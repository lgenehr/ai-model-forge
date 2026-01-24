import pytest
from src.pipeline.guardrails import Guardrails

def test_guardrails_financial_advice():
    g = Guardrails()
    # Should block
    assert g.check("This is a strong buy now") is False
    assert g.check("We are going to the moon") is False
    assert g.check("Pump it up") is False
    assert g.check("Price target of $500") is False
    
    # Should pass
    assert g.check("The stock market closed higher today.") is True
    assert g.check("Bitcoin is a cryptocurrency.") is True

def test_guardrails_harmful():
    g = Guardrails()
    assert g.check("I hate everyone") is False
    assert g.check("Love is good") is True
