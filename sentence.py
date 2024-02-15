
import sentencepiece as spm
# Load the existing sp.model file
model_path = '/home/neko/Downloads/sp.model'




# Load the existing sp.model file

sp = spm.SentencePieceProcessor()
sp.Load(model_path)

# Get the vocabulary
vocabulary = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]

# Save the vocabulary as a text file
output_file = 'path_to_output.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    file.write('\n'.join(vocabulary))

print("Vocabulary saved as text file:", output_file)
