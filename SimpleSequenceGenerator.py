import tflearn
import re
import numpy as np

def fade(x):

	return x / 2



class SimpleSequenceGenerator(tflearn.DNN):

	def __init__(self, vocab):

		self.vocab = vocab
		self.vocab_size = len(vocab)
		self.idx_freq = np.zeros(self.vocab_size)
		self.char_idx = {j:i for i, j in enumerate(vocab)}
		self.idx_char = {i:j for i, j in enumerate(vocab)}
		# Creating the model
		model = tflearn.input_data(shape = [None, len(self.vocab)], name = 'inputs')
		model = tflearn.fully_connected(model, 2 * len(vocab), activation = 'relu')
		model = tflearn.dropout(model, 0.5)
		model = tflearn.fully_connected(model, 2 * len(vocab), activation = 'relu')
		model = tflearn.dropout(model, 0.5)
		model = tflearn.fully_connected(model, len(vocab), activation = 'softmax')
		model = tflearn.regression(model, optimizer = 'adam', loss = 'categorical_crossentropy', learning_rate = 0.001)
		tflearn.DNN.__init__(self, model)

	def one_hot(self, n):
		a = np.zeros(self.vocab_size)
		a[n] = 1
		return a

	def seperate_sentences(self, text):
		sentences = text.split('\n')
		return [i.split() for i in sentences]

	def fit(self, text, n_epoch = 3):

		x = []
		y = []
		sentences = self.seperate_sentences(text)
		for sentence in sentences:
			current_x = np.zeros(self.vocab_size)
			for i in range(len(sentence) - 1):
				idx = self.char_idx[sentence[i]]
				self.idx_freq[idx] += 1
				current_x = fade(current_x)
				current_x[idx] += 1
				x.append(np.array(current_x))
				y.append(self.one_hot(self.char_idx[sentence[i + 1]]))

		print('Training data: {} rows'.format(len(x)))
		x = np.array(x)
		y = np.array(y)
		tflearn.DNN.fit(self, x, y, n_epoch = n_epoch)

	def generate(self, seed, gen_len = 20):
		seed_value = np.zeros(self.vocab_size)
		seed_words = seed.split(' ')
		for word in seed_words:
			seed_value = fade(seed_value)
			seed_value[self.char_idx[word]] += 1

		fade_val = 0
		for i in range(gen_len):
			result = self.predict([seed_value])
			result_idx = np.random.choice(self.vocab_size, p = result[0])
			#result_idx = np.argmax(result[0])
			if result[0][result_idx] < 2 / self.vocab_size:
				return seed
			seed_value = fade(seed_value)
			seed += ' ' + self.idx_char[result_idx]
			seed_value[result_idx] += 1

		return seed

def strip_unknown_chars(text):
	text = text.lower()
	text = re.sub('[^a-z \n]', '', text)
	return text

text = ''
with open('headlines.txt', 'r') as f:
	limit = 1500
	i = 0
	for line in f:
		if 'syria' in line:
			text += line + '\n'
			i += 1
			if i == limit:
				break
text = text[:-1]
text = strip_unknown_chars(text)
words = list(set(text.split()))

model = SimpleSequenceGenerator(words)
model.fit(text, 40)

new_lines = []
for i in range(50):
	np.random.shuffle(words)
	seed = words[0] + ' ' + words[1]
	sentence = model.generate(seed, 20)
	with open('generated_text.txt', 'a+') as f:
		f.write(sentence + '\n')
