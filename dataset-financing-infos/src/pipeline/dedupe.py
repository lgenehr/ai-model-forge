import hashlib

class Deduplicator:
    def __init__(self):
        self.seen_hashes = set()

    def is_duplicate(self, text: str) -> bool:
        """
        Returns True if text is a duplicate of something already seen.
        """
        if not text:
            return True # Empty is 'duplicate' or 'bad'
            
        # Create a hash of the content
        # We strip spaces to be slightly robust against formatting differences
        content_hash = hashlib.md5(text.replace(" ", "").encode('utf-8')).hexdigest()
        
        if content_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(content_hash)
        return False
