from llm import LLMClient

llm_client = LLMClient()
def test_generate_response():
    prompt = "Hello, how are you?"
    response = llm_client.generate_response(prompt)
    assert isinstance(response, str)
    assert len(response) > 0
    
test_generate_response()