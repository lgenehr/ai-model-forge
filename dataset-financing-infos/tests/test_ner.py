from src.pipeline.ner import MetadataExtractor

def test_extract_topics():
    # We need a mock config or ensure the default one loads
    # For unit test, we can mock the loader or just use a small config
    extractor = MetadataExtractor()
    extractor.keyword_map = {
        "crypto": ["bitcoin", "blockchain"],
        "tech": ["AI", "computer"]
    }
    
    topics = extractor.extract_topics("Bitcoin is a blockchain technology.")
    assert "crypto" in topics
    assert "tech" not in topics

    topics = extractor.extract_topics("AI computer.")
    assert "tech" in topics

def test_extract_entities():
    extractor = MetadataExtractor()
    text = "The United States and Apple Inc are huge."
    entities = extractor.extract_entities(text)
    assert "United States" in entities
    assert "Apple Inc" in entities
    assert "The" not in entities # Assuming we filter simple starts or short words
