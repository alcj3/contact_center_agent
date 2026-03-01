from app.classifier import IntentClassifier


def test_classifier_stub_interface() -> None:
    classifier = IntentClassifier()
    prediction = classifier.predict("I have a question")

    assert isinstance(prediction.intent, str)
    assert isinstance(prediction.confidence, float)
