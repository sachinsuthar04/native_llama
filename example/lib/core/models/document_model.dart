enum DocumentType { text, audio, image, pdf }

class DocumentModel {
  final int? id;
  final String title;
  final String content;
  final DocumentType type;
  final DateTime createdAt;

  DocumentModel({
    this.id,
    required this.title,
    required this.content,
    required this.type,
    required this.createdAt,
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'title': title,
      'content': content,
      'type': type.index,
      'createdAt': createdAt.toIso8601String(),
    };
  }

  factory DocumentModel.fromMap(Map<String, dynamic> map) {
    return DocumentModel(
      id: map['id'],
      title: map['title'],
      content: map['content'],
      type: DocumentType.values[map['type']],
      createdAt: DateTime.parse(map['createdAt']),
    );
  }
}
