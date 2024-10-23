# Copyright (C) 2024 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


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
