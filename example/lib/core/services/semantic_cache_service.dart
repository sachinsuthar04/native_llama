import 'dart:math';
import 'database_service.dart';
import 'llama_service.dart';

class SemanticCacheService {
  final DatabaseService _db = DatabaseService.instance;
  final LlamaService _llama = LlamaService.instance;
  static const double threshold = 0.85;

  Future<String?> getCachedResponse(String query) async {
    final embedding = await _llama.getEmbedding(query);
    if (embedding.isEmpty) return null;

    final entries = await _db.getCacheEntries();
    double maxSimilarity = -1.0;
    String? bestResponse;

    for (final entry in entries) {
      final cachedEmbedding = entry['embedding'] as List<double>;
      final similarity = _cosineSimilarity(embedding, cachedEmbedding);

      if (similarity > maxSimilarity) {
        maxSimilarity = similarity;
        bestResponse = entry['response'];
      }
    }

    if (maxSimilarity >= threshold) {
      print('Semantic Cache HIT: $maxSimilarity');
      return bestResponse;
    }
    return null;
  }

  Future<void> cacheResponse(String query, String response) async {
    final embedding = await _llama.getEmbedding(query);
    if (embedding.isNotEmpty) {
      await _db.saveToCache(query, response, embedding);
    }
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) return 0.0;
    double dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA == 0 || normB == 0) return 0.0;
    return dotProduct / (sqrt(normA) * sqrt(normB));
  }
}