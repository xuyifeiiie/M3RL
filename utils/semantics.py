# feature_extractor.py
import numpy as np
import os
import re
import random
from typing import Tuple, Dict, List, Optional, Any
from gensim.models import FastText
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import logging  # Keep logging if you prefer it over print for OOV
from utils.perturb import PerturbationManager

import time
import json
import psutil
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA


# --- Tokenize function (from your code) ---
def tokenize(log):
    word_list_tmp = re.findall(r"[a-zA-Z]+", log)
    word_list = []
    for word in word_list_tmp:
        res = list(filter(None, re.split("([A-Z][a-z][^A-Z]*)", word)))
        if len(res) == 0:
            word_list.append(word.lower())
        else:
            word_list.extend([w.lower() for w in res])
    return word_list


class Vocab:
    def __init__(self, **kwargs):
        self.embedding_dim = kwargs["word_embedding_dim"]
        self.save_dir = kwargs["word2vec_save_dir"]
        self.model_type = kwargs["word2vec_model_type"]
        self.epochs = kwargs["word2vec_epoch"]
        self.word_window = kwargs["word_window"]
        self.log_length = 0
        self.model_save_path = os.path.join(
            self.save_dir, f"{self.model_type}-{self.embedding_dim}d.model"
        )
        self.corpus_save_path = os.path.join(
            self.save_dir, f"{self.model_type}-{self.embedding_dim}d_corpus.pkl"
        )
        self.word2vec_model = None
        self.wv = None
        self.known_words = None

    def train_or_load_model(self, logs_for_training_flat_list):
        corpus_for_tfidf = []
        if os.path.exists(self.model_save_path):
            print(
                f"Loading existing {self.model_type} model from {self.model_save_path}"
            )
            if self.model_type == "fasttext":
                self.word2vec_model = FastText.load(self.model_save_path)
            else:
                raise NotImplementedError(
                    f"Model type {self.model_type} loading not supported."
                )
            if os.path.exists(self.corpus_save_path):
                with open(self.corpus_save_path, "rb") as f:
                    corpus_for_tfidf = pickle.load(f)
            else:
                print(
                    f"Warning: Model exists but corpus PKL {self.corpus_save_path} not found. Rebuilding."
                )
                for log_text in logs_for_training_flat_list:
                    tokenized_log = tokenize(log_text)
                    if tokenized_log:
                        corpus_for_tfidf.append(" ".join(tokenized_log))
                if corpus_for_tfidf:
                    with open(self.corpus_save_path, "wb") as f:
                        pickle.dump(corpus_for_tfidf, f)
        else:
            print(f"Training new {self.model_type} model...")
            tokenized_sentences = []
            for (
                log_text
            ) in logs_for_training_flat_list:  # Expects flat list of log strings
                word_list = tokenize(log_text)
                if not word_list:
                    continue
                self.log_length = max(self.log_length, len(word_list))
                tokenized_sentences.append(word_list)
                corpus_for_tfidf.append(" ".join(word_list))
            if not tokenized_sentences:
                raise ValueError("No valid logs to train word embedding model.")
            if self.model_type == "fasttext":
                self.word2vec_model = FastText(
                    sentences=tokenized_sentences,
                    window=self.word_window,
                    min_count=1,
                    vector_size=self.embedding_dim,
                    epochs=self.epochs,
                    workers=max(1, os.cpu_count() - 1 if os.cpu_count() else 1),
                )  # Ensure workers >= 1
            else:
                raise NotImplementedError(
                    f"Model type {self.model_type} training not supported."
                )
            os.makedirs(self.save_dir, exist_ok=True)
            self.word2vec_model.save(self.model_save_path)
            print(f"Model saved to {self.model_save_path}")
            with open(self.corpus_save_path, "wb") as f:
                pickle.dump(corpus_for_tfidf, f)
            print(f"Corpus for TF-IDF saved to {self.corpus_save_path}")
        self.wv = self.word2vec_model.wv
        self.known_words = self.wv.key_to_index
        return corpus_for_tfidf


class FeatureExtractor:
    def __init__(self, settings, **kwargs):
        self.settings = settings
        self.history_steps = settings.history_steps
        self.num_nodes = settings.num_nodes
        self.vocab = Vocab(**kwargs)
        self.embedding_dim = self.vocab.embedding_dim
        self.meta_data = {"num_labels": 2, "max_log_length": 1}
        self.oov = set()

        self.if_tfidf = kwargs.get("if_tfidf", False)

        self.model_type = kwargs.get("word2vec_model_type", "fasttext")
        self.bert_model_name = kwargs.get("bert_model_name", "bert-base-uncased")
        self.bert_batch_size = kwargs.get("bert_batch_size", 64)
        self.bert_max_length = kwargs.get("bert_max_length", 64)
        self.profile_log_encoding = kwargs.get("profile_log_encoding", True)
        self.device = kwargs.get("device", "cpu")
        if isinstance(self.device, str):
            self.device = torch.device(
                self.device
                if torch.cuda.is_available() or "cpu" in self.device
                else "cpu"
            )

        # BERT-related cache.
        self.bert_tokenizer = None
        self.bert_model = None
        self.bert_hidden_dim = None

        # PCA-related cache.
        self.pca_target_dim = kwargs.get("pca_target_dim", self.embedding_dim)
        self.pca_model_path = os.path.join(
            self.vocab.save_dir,
            f"{self.bert_model_name.replace('/', '_')}_pca_{self.pca_target_dim}.pkl",
        )
        self.pca_model = None

        self.profile_save_path = os.path.join(
            self.vocab.save_dir, f"{self.model_type}_encoding_profile.jsonl"
        )

        self.tfidf_vectorizer_path = os.path.join(
            self.vocab.save_dir, "tfidf_vectorizer_main.pkl"
        )
        self.tfidf_transformer_path = os.path.join(
            self.vocab.save_dir, "tfidf_transformer_main.pkl"
        )
        self.count_vectorizer_fitted = None
        self.tfidf_transformer_fitted = None
        print(
            f"FeatureExtractor initialized. Word embedding: {self.vocab.model_type}, TF-IDF: {self.if_tfidf}"
        )

    def _get_cpu_mem_mb(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _start_profile(self):
        profile = {}
        profile["start_time"] = time.perf_counter()
        profile["cpu_mem_before_mb"] = self._get_cpu_mem_mb()

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            profile["gpu_mem_before_mb"] = (
                torch.cuda.memory_allocated(self.device) / 1024 / 1024
            )
        else:
            profile["gpu_mem_before_mb"] = 0.0
        return profile

    def _end_profile(self, profile, num_logs, split_name):
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        elapsed_sec = time.perf_counter() - profile["start_time"]
        cpu_mem_after_mb = self._get_cpu_mem_mb()
        cpu_mem_delta_mb = cpu_mem_after_mb - profile["cpu_mem_before_mb"]

        if torch.cuda.is_available() and self.device.type == "cuda":
            gpu_mem_after_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            gpu_mem_delta_mb = gpu_mem_after_mb - profile["gpu_mem_before_mb"]
            gpu_peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        else:
            gpu_mem_after_mb = 0.0
            gpu_mem_delta_mb = 0.0
            gpu_peak_mb = 0.0

        stats = {
            "split": split_name,
            "encoder": self.model_type,
            "num_logs": int(num_logs),
            "elapsed_sec": float(elapsed_sec),
            "ms_per_log": float(elapsed_sec * 1000 / max(num_logs, 1)),
            "logs_per_sec": float(num_logs / max(elapsed_sec, 1e-8)),
            "cpu_mem_before_mb": float(profile["cpu_mem_before_mb"]),
            "cpu_mem_after_mb": float(cpu_mem_after_mb),
            "cpu_mem_delta_mb": float(cpu_mem_delta_mb),
            "gpu_mem_before_mb": float(profile["gpu_mem_before_mb"]),
            "gpu_mem_after_mb": float(gpu_mem_after_mb),
            "gpu_mem_delta_mb": float(gpu_mem_delta_mb),
            "gpu_peak_mb": float(gpu_peak_mb),
        }

        print("[LOG_ENCODING_PROFILE] " + json.dumps(stats, ensure_ascii=False))

        with open(self.profile_save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")

        return stats

    def _load_bert_if_needed(self):
        if self.bert_tokenizer is None or self.bert_model is None:
            print(f"Loading BERT model: {self.bert_model_name}")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
            self.bert_model.eval()
            self.bert_model.to(self.device)
            self.bert_hidden_dim = self.bert_model.config.hidden_size

    def _load_pca_if_exists(self):
        if self.pca_model is None and os.path.exists(self.pca_model_path):
            print(f"Loading PCA from {self.pca_model_path}")
            with open(self.pca_model_path, "rb") as f:
                self.pca_model = pickle.load(f)

    def _fit_pca_if_needed(self, train_embeddings):
        if self.pca_model is None:
            print(f"Fitting PCA: {train_embeddings.shape[1]} -> {self.pca_target_dim}")
            self.pca_model = PCA(n_components=self.pca_target_dim, random_state=42)
            self.pca_model.fit(train_embeddings)
            os.makedirs(self.vocab.save_dir, exist_ok=True)
            with open(self.pca_model_path, "wb") as f:
                pickle.dump(self.pca_model, f)
            print(f"PCA saved to {self.pca_model_path}")

    @torch.no_grad()
    def _encode_logs_with_bert_batch(self, texts):
        self._load_bert_if_needed()
        if len(texts) == 0:
            return np.zeros((0, self.bert_hidden_dim), dtype=np.float32)

        all_embeddings = []
        for start_idx in range(0, len(texts), self.bert_batch_size):
            batch_texts = texts[start_idx : start_idx + self.bert_batch_size]

            encoded = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.bert_max_length,
                return_tensors="pt",
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.bert_model(**encoded)
            last_hidden_state = outputs.last_hidden_state  # [B, L, H]
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # [B, L, 1]

            # mean pooling
            masked_hidden = last_hidden_state * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)  # [B, H]
            valid_token_count = attention_mask.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / valid_token_count  # [B, H]
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embeddings, axis=0)

    def _collect_all_logs_with_positions(self, raw_chunks_dict_input):
        all_logs = []
        all_positions = []  # [(chunk_id, ts_key, node_key)]

        for chunk_id, item_original in raw_chunks_dict_input.items():
            original_logs_dict_text = item_original.get("logs", {})
            for ts_key in range(self.history_steps):
                node_logs_at_ts = original_logs_dict_text.get(ts_key, {})
                for node_key in range(self.num_nodes):
                    log_list_text_for_node = node_logs_at_ts.get(node_key, [])
                    for log_str in log_list_text_for_node:
                        if isinstance(log_str, str) and len(tokenize(log_str)) > 0:
                            all_logs.append(log_str)
                            all_positions.append((chunk_id, ts_key, node_key))
        return all_logs, all_positions

    def _transform_with_bert_pca(self, raw_chunks_dict_input, datatype="train"):
        print(
            f"FeatureExtractor._transform_with_bert_pca() for '{datatype}' data ({len(raw_chunks_dict_input)} chunks)."
        )

        # 1. Initialize output containers.
        output_chunks_with_emb = {}
        for chunk_id, item_original in raw_chunks_dict_input.items():
            item_transformed = {k: v for k, v in item_original.items() if k != "logs"}
            if "log_stats" not in item_transformed and "log_stats" in item_original:
                item_transformed["log_stats"] = item_original["log_stats"]
            item_transformed["log_emb"] = np.zeros(
                (self.history_steps, self.num_nodes, self.pca_target_dim),
                dtype=np.float32,
            )
            output_chunks_with_emb[chunk_id] = item_transformed

        # 2. Collect logs and their target positions.
        all_logs, all_positions = self._collect_all_logs_with_positions(
            raw_chunks_dict_input
        )
        num_logs = len(all_logs)
        print(f"  Collected {num_logs} logs for BERT encoding on split '{datatype}'.")

        if num_logs == 0:
            return output_chunks_with_emb, {"num_logs": 0}

        # 3. Profile only the embedding stage.
        profile = self._start_profile()

        bert_embeddings = self._encode_logs_with_bert_batch(all_logs)  # [num_logs, H]

        # 4. PCA
        if datatype == "train":
            self._fit_pca_if_needed(bert_embeddings)
        else:
            self._load_pca_if_exists()
            if self.pca_model is None:
                raise RuntimeError(
                    "PCA model not found for non-train split. Please process train split first."
                )

        projected_embeddings = self.pca_model.transform(bert_embeddings).astype(
            np.float32
        )  # [num_logs, D]

        profile_stats = self._end_profile(profile, num_logs, datatype)

        # 5. Scatter embeddings back to each (chunk, time, node).
        temp_bucket = {}
        for emb, pos in zip(projected_embeddings, all_positions):
            temp_bucket.setdefault(pos, []).append(emb)

        for (chunk_id, ts_key, node_key), emb_list in temp_bucket.items():
            output_chunks_with_emb[chunk_id]["log_emb"][ts_key][node_key] = np.mean(
                emb_list, axis=0
            ).astype(np.float32)

        return output_chunks_with_emb

    def fasttext_model_exists(self):
        return os.path.exists(self.vocab.model_save_path)

    def tfidf_models_exist(self):
        if not self.if_tfidf:
            return True
        return os.path.exists(self.tfidf_vectorizer_path) and os.path.exists(
            self.tfidf_transformer_path
        )

    def _load_models_if_needed(self):
        if self.vocab.wv is None:
            if self.fasttext_model_exists():
                print(
                    f"Loading FastText WV from {self.vocab.model_save_path} for transform..."
                )
                self.vocab.train_or_load_model([])
            else:
                raise FileNotFoundError(
                    f"FastText model not found at {self.vocab.model_save_path}"
                )

        if self.if_tfidf:
            if self.count_vectorizer_fitted is None and os.path.exists(
                self.tfidf_vectorizer_path
            ):
                print(f"Loading CountVectorizer from {self.tfidf_vectorizer_path}")
                with open(self.tfidf_vectorizer_path, "rb") as f:
                    self.count_vectorizer_fitted = pickle.load(f)
            if self.tfidf_transformer_fitted is None and os.path.exists(
                self.tfidf_transformer_path
            ):
                print(f"Loading TfidfTransformer from {self.tfidf_transformer_path}")
                with open(self.tfidf_transformer_path, "rb") as f:
                    self.tfidf_transformer_fitted = pickle.load(f)
            if not self.count_vectorizer_fitted or not self.tfidf_transformer_fitted:
                raise FileNotFoundError(
                    "TF-IDF models (CountVectorizer or TfidfTransformer) not found for transform, but if_tfidf is True."
                )

    def __get_word_vector(self, word):
        # ***** [updated code] # ****** Make sure self.vocab.known_words and self.vocab.wv are loaded
        if self.vocab.known_words is None or self.vocab.wv is None:
            self._load_models_if_needed()  # Ensure models are loaded before accessing wv/known_words

        if word in self.vocab.known_words:
            return self.vocab.wv[word]
        else:
            self.oov.add(word)
            if self.vocab.model_type == "fasttext":
                return self.vocab.wv[word]
            else:
                return np.zeros(self.embedding_dim, dtype=np.float32)

    # ***** [updated code] # ******
    def __seq_to_emb(self, single_log_text, single_log_tfidf_vector=None):
        """
        Converts a single log message string to an embedding.
        single_log_tfidf_vector: A 1D NumPy array (dense) or a SciPy sparse matrix row (1, vocab_size)
                                 representing TF-IDF weights for words in THIS log, aligned with
                                 self.count_vectorizer_fitted.vocabulary_.
        """
        word_list = tokenize(single_log_text)
        if not word_list:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        log_embedding_sum = np.zeros(self.embedding_dim, dtype=np.float32)
        sum_of_weights = 0.0

        if (
            self.if_tfidf
            and single_log_tfidf_vector is not None
            and self.count_vectorizer_fitted
        ):
            # tfidf_vocab: word -> column_index in the TF-IDF vector
            tfidf_vocab = self.count_vectorizer_fitted.vocabulary_

            for word in word_list:
                word_vec = self.__get_word_vector(word)  # Shape: (embedding_dim,)
                weight = 1.0  # Default weight if word not in TF-IDF vocab or no TF-IDF

                if word in tfidf_vocab:
                    word_idx_in_tfidf = tfidf_vocab[word]
                    # single_log_tfidf_vector is for the current log.
                    # If it's a sparse matrix row (e.g., from csr_matrix.getrow()), it's (1, vocab_size)
                    # If it's dense, it should be (vocab_size,)
                    try:
                        if hasattr(
                            single_log_tfidf_vector, "tocsr"
                        ):  # Check if it's a sparse matrix/row
                            if word_idx_in_tfidf < single_log_tfidf_vector.shape[1]:
                                weight = single_log_tfidf_vector[
                                    0, word_idx_in_tfidf
                                ]  # Access sparse matrix
                        elif isinstance(
                            single_log_tfidf_vector, np.ndarray
                        ):  # Dense array
                            if word_idx_in_tfidf < single_log_tfidf_vector.shape[0]:
                                weight = single_log_tfidf_vector[word_idx_in_tfidf]
                        else:  # Fallback or unrecognized type
                            pass  # weight remains 1.0

                    except (
                        IndexError
                    ):  # Should not happen if vocab and vector are aligned
                        pass  # weight remains 1.0

                log_embedding_sum += (
                    weight * word_vec
                )  # weight is scalar, word_vec is (emb_dim,)
                sum_of_weights += weight

            return (
                log_embedding_sum / sum_of_weights
                if sum_of_weights > 0.0
                else np.zeros(self.embedding_dim, dtype=np.float32)
            )
        else:  # No TF-IDF, or TF-IDF vector not provided for this log
            embeddings = [self.__get_word_vector(word) for word in word_list]
            if not embeddings:
                return np.zeros(self.embedding_dim, dtype=np.float32)
            return np.mean(embeddings, axis=0).astype(np.float32)

    # ***** [updated code] # ******

    def fit(
        self, data_reader
    ):  # data_reader is ReadHDF5MultiModalData (for preprocessing)
        print("FeatureExtractor.fit starting...")
        all_train_log_texts = data_reader.train_logs

        if not all_train_log_texts:
            print(
                "No training logs provided to FeatureExtractor.fit(). Skipping Word2Vec/FastText training."
            )
            corpus_for_tfidf_training = []  # Define it as empty if no logs
        else:
            corpus_for_tfidf_training = self.vocab.train_or_load_model(
                all_train_log_texts
            )
            self.meta_data["vocab_size"] = (
                len(self.vocab.known_words) if self.vocab.known_words else 0
            )
            self.meta_data["max_log_length"] = (
                self.vocab.log_length if self.vocab.log_length > 0 else 100
            )
            print(
                f"  Vocabulary size: {self.meta_data['vocab_size']}, Max log token length: {self.meta_data['max_log_length']}"
            )

        if self.if_tfidf:
            if not self.tfidf_models_exist():
                print("TF-IDF models not found. Training TF-IDF...")
                if not corpus_for_tfidf_training:
                    print(
                        "  Corpus for TF-IDF is empty (from vocab step). Skipping TF-IDF training."
                    )
                else:
                    print(
                        f"  Fitting CountVectorizer on {len(corpus_for_tfidf_training)} processed log strings..."
                    )
                    # Use a tokenizer that splits on non-alphanumeric to better match `tokenize` if needed,
                    # or ensure `corpus_for_tfidf_training` is list of already tokenized-and-space-joined strings.
                    self.count_vectorizer_fitted = CountVectorizer(
                        lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]*\b"
                    )
                    term_counts_train = self.count_vectorizer_fitted.fit_transform(
                        corpus_for_tfidf_training
                    )

                    print(f"  Fitting TfidfTransformer...")
                    self.tfidf_transformer_fitted = TfidfTransformer()
                    _ = self.tfidf_transformer_fitted.fit_transform(term_counts_train)

                    os.makedirs(self.vocab.save_dir, exist_ok=True)
                    with open(self.tfidf_vectorizer_path, "wb") as f:
                        pickle.dump(self.count_vectorizer_fitted, f)
                    with open(self.tfidf_transformer_path, "wb") as f:
                        pickle.dump(self.tfidf_transformer_fitted, f)
                    print(
                        f"  TF-IDF CountVectorizer saved to {self.tfidf_vectorizer_path}"
                    )
                    print(
                        f"  TF-IDF Transformer saved to {self.tfidf_transformer_path}"
                    )
            else:
                print("TF-IDF models found. Skipping TF-IDF training.")
        print("FeatureExtractor.fit() complete.")

    def _perturb_log_vector_list(
        self,
        log_embedding_vector_list: List[np.ndarray],
        log_perturb_method: str,
        log_perturb_intensity: float,
    ) -> List[np.ndarray]:
        """
        Applies delete/duplicate perturbations to a list of log embedding vectors.
        """
        if (
            log_perturb_method == "none"
            or log_perturb_intensity <= 0
            or not log_embedding_vector_list
        ):
            return log_embedding_vector_list

        perturbed_list = list(log_embedding_vector_list)  # Shallow copy
        num_log_vectors = len(perturbed_list)
        num_to_affect = int(num_log_vectors * log_perturb_intensity)
        if num_to_affect == 0 and num_log_vectors > 0 and log_perturb_intensity > 0:
            num_to_affect = 1
        num_to_affect = min(
            num_to_affect, num_log_vectors
        )  # Cannot affect more than available

        if log_perturb_method == "delete":
            if num_log_vectors > 0:
                for _ in range(num_to_affect):
                    if perturbed_list:
                        del_idx = random.randint(0, len(perturbed_list) - 1)
                        perturbed_list.pop(del_idx)
        elif log_perturb_method == "duplicate":
            if num_log_vectors > 0:
                for _ in range(num_to_affect):
                    if not perturbed_list:
                        break
                    dup_idx = random.randint(0, len(perturbed_list) - 1)
                    vector_to_duplicate = perturbed_list[dup_idx].copy()  # Copy ndarray
                    insert_pos = random.randint(0, len(perturbed_list))
                    perturbed_list.insert(insert_pos, vector_to_duplicate)
        return perturbed_list

    def _transform_with_fasttext(self, raw_chunks_dict_input, datatype="train"):
        print(
            f"FeatureExtractor.transform() for '{datatype}' data ({len(raw_chunks_dict_input)} chunks)."
        )
        self._load_models_if_needed()
        self.oov = set()
        output_chunks_with_emb = {}

        current_split_tfidf_matrix = None
        logs_fed_to_tfidf = []

        if (
            self.if_tfidf
            and self.count_vectorizer_fitted
            and self.tfidf_transformer_fitted
        ):
            print(
                f"  Preparing all logs from '{datatype}' split for TF-IDF transformation..."
            )
            temp_logs_for_tfidf_vectorization = []
            for chunk_id, item_original in raw_chunks_dict_input.items():
                original_logs_dict_text = item_original.get("logs", {})
                for ts_key in range(self.history_steps):
                    node_logs_at_ts = original_logs_dict_text.get(ts_key, {})
                    for node_key in range(self.num_nodes):
                        log_list_text_for_node = node_logs_at_ts.get(node_key, [])
                        for log_str in log_list_text_for_node:
                            tokenized_log_for_tfidf = tokenize(log_str)
                            if tokenized_log_for_tfidf:
                                temp_logs_for_tfidf_vectorization.append(
                                    " ".join(tokenized_log_for_tfidf)
                                )
                                logs_fed_to_tfidf.append(log_str)

            if temp_logs_for_tfidf_vectorization:
                print(
                    f"  Transforming {len(temp_logs_for_tfidf_vectorization)} logs with TF-IDF for '{datatype}' split..."
                )
                term_counts_current_split = self.count_vectorizer_fitted.transform(
                    temp_logs_for_tfidf_vectorization
                )
                current_split_tfidf_matrix = self.tfidf_transformer_fitted.transform(
                    term_counts_current_split
                )
                print(
                    f"  TF-IDF transformation complete for {datatype}. Matrix shape: {current_split_tfidf_matrix.shape}"
                )
            else:
                print(f"  No logs to transform with TF-IDF for {datatype}.")

        target_mod = self.settings.raw_data_target_modality
        flat_log_idx_counter = 0

        # Count logs once before semantic encoding.
        num_logs_for_encoding = 0
        for chunk_id, item_original in raw_chunks_dict_input.items():
            original_logs_dict_text = item_original.get("logs", {})
            for ts_key in range(self.history_steps):
                node_logs_at_ts = original_logs_dict_text.get(ts_key, {})
                for node_key in range(self.num_nodes):
                    log_list_text_for_node = node_logs_at_ts.get(node_key, [])
                    for log_str in log_list_text_for_node:
                        if tokenize(log_str):
                            num_logs_for_encoding += 1

        # Start profiling after the cheap counting pass.
        profile = self._start_profile()

        for chunk_id, item_original in raw_chunks_dict_input.items():
            node_mark = 0
            labels = item_original["labels"]
            item_transformed = {k: v for k, v in item_original.items() if k != "logs"}
            if "log_stats" not in item_transformed and "log_stats" in item_original:
                item_transformed["log_stats"] = item_original["log_stats"]

            log_emb_tensor = np.zeros(
                (self.history_steps, self.num_nodes, self.embedding_dim),
                dtype=np.float32,
            )
            original_logs_dict_text = item_original.get("logs", {})

            for ts_key in range(self.history_steps):
                node_logs_at_ts = original_logs_dict_text.get(ts_key, {})
                for node_key in range(self.num_nodes):
                    log_list_text_for_node = node_logs_at_ts.get(node_key, [])

                    if log_list_text_for_node:
                        node_log_embeddings_list = []
                        for single_log_str in log_list_text_for_node:
                            single_log_tfidf_vec = None
                            if self.if_tfidf and current_split_tfidf_matrix is not None:
                                if tokenize(single_log_str):
                                    if (
                                        flat_log_idx_counter
                                        < current_split_tfidf_matrix.shape[0]
                                    ):
                                        single_log_tfidf_vec = (
                                            current_split_tfidf_matrix.getrow(
                                                flat_log_idx_counter
                                            )
                                        )
                                        flat_log_idx_counter += 1

                            node_log_embeddings_list.append(
                                self.__seq_to_emb(single_log_str, single_log_tfidf_vec)
                            )

                        if (
                            datatype == "test"
                            and self.settings.if_perturb
                            and target_mod == "logs"
                            and labels == -1
                            and node_log_embeddings_list
                        ):
                            random_pertub_rate = random.random()
                            node_mark = node_key
                            if (
                                random_pertub_rate
                                < self.settings.raw_data_log_list_perturb_sample_rate
                            ):
                                print(
                                    f"  Applying {self.settings.log_perturb_method} perturbations on raw logs to TEST split data..."
                                )
                                node_log_embeddings_list = (
                                    self._perturb_log_vector_list(
                                        node_log_embeddings_list,
                                        self.settings.log_perturb_method,
                                        self.settings.log_perturb_intensity,
                                    )
                                )

                        if node_log_embeddings_list:
                            log_emb_tensor[ts_key][node_key] = np.mean(
                                node_log_embeddings_list, axis=0
                            )

            item_transformed["log_emb"] = log_emb_tensor
            if datatype == "test" and self.settings.if_perturb and target_mod == "logs":
                item_transformed["labels"] = node_mark
                print(
                    f"      Label changed due to log list perturbation: -1 -> {node_mark}"
                )
            output_chunks_with_emb[chunk_id] = item_transformed

        if len(self.oov) > 0:
            logging.info(
                f"{datatype} split: {len(self.oov)} OOV words: {','.join(list(self.oov)[:10])}..."
            )

        if (
            self.if_tfidf
            and current_split_tfidf_matrix is not None
            and flat_log_idx_counter != len(logs_fed_to_tfidf)
        ):
            print(
                f"Warning: TF-IDF log count mismatch for {datatype}. Processed {flat_log_idx_counter} TF-IDF vectors, but collected {len(logs_fed_to_tfidf)} logs for TF-IDF."
            )

        _ = self._end_profile(profile, num_logs_for_encoding, datatype)

        return output_chunks_with_emb

    def transform(self, raw_chunks_dict_input, datatype="train"):
        if self.model_type == "bert":
            output_chunks_with_emb = self._transform_with_bert_pca(
                raw_chunks_dict_input, datatype
            )
            return output_chunks_with_emb
        else:
            return self._transform_with_fasttext(raw_chunks_dict_input, datatype)
