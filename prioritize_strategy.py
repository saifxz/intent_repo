from abc import ABC, abstractmethod

class PriorityStrategy(ABC):
    @abstractmethod
    def get_priority(self, message: dict) -> int:
        pass

class SentimentPriorityStrategy(PriorityStrategy):
    def get_priority(self, message: dict) -> int:
        sentiment = message.get("sentiment", "Neutral")

        mapping = {
            "Negative": 3,
            "Neutral": 2,
            "Positive": 1
        }

        return mapping.get(sentiment, 1)
    
class IntentPriorityStrategy(PriorityStrategy):
    def get_priority(self, message: dict) -> int:
        intent = message.get("category", "")

        high_priority_intents = ["refund", "complaint", "fraud"]

        if intent in high_priority_intents:
            return 3
        return 1

class HybridPriorityStrategy(PriorityStrategy):
    def get_priority(self, message: dict) -> int:
        sentiment = message.get("sentiment", "Neutral")
        intent = message.get("category", "")

        if sentiment == "Negative":
            return 3
        if intent in ["refund", "complaint"]:
            return 3
        if sentiment == "Neutral":
            return 2
        return 1