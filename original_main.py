
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# We should rename this
class AI_Test:
    def __init__(self):
        pass

    def generate_bot_response(self, question):
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")

        prompt = '''
        Your task is to answer children's questions using simple language.
        Explain any difficult words in a way a 3-year-old can understand.
        Keep responses under 60 words.
        \n\nQuestion:
        '''

        input_text = prompt + question

        inputs = tokenizer.encode(input_text, return_tensors='pt')
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer
