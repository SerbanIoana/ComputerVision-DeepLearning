# load doc into memory
import os
import string


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# extract captions for images
def load_captions(doc):
    descriptions = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        img_id, img_caption = tokens[0], tokens[1:]
        # remove filename from image id
        img_id = img_id.split('.')[0]
        # convert description tokens back to string
        img_caption = ' '.join(img_caption)
        # create the list if needed
        if img_id not in descriptions:
            descriptions[img_id] = list()
        # store description
        descriptions[img_id].append(img_caption)
    return descriptions


def clean_captions(captions):
    # prepare translation translator for removing punctuation
    translator = str.maketrans('', '', string.punctuation)
    for img_id, captions_list in captions.items():
        for i in range(len(captions_list)):
            caption = captions_list[i]
            # tokenize
            caption = caption.split()
            # convert to lower case
            caption = [word.lower() for word in caption]
            # remove punctuation from each token
            caption = [word.translate(translator) for word in caption]
            # remove all words that are one character or less in length eg. 's' and 'a'
            caption = [word for word in caption if len(word) > 1]
            # remove tokens with numbers in them
            caption = [word for word in caption if word.isalpha()]
            # store as string
            captions_list[i] = ' '.join(caption)


# convert the loaded captions into a vocabulary of words
def create_vocabulary(descriptions):
    # build a list of all unique words from image captions
    vocabulary = set()
    for key in descriptions.keys():
        [vocabulary.update(d.split()) for d in descriptions[key]]
    return vocabulary


# save captions to file, one per line
def save_captions(descriptions, filename):
    lines = list()
    for img_id, captions_list in descriptions.items():
        for caption in captions_list:
            lines.append(img_id + ' ' + caption)
    all_captions = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(all_captions)
    file.close()


if __name__ == "__main__":
    filename = os.path.dirname(__file__) + '\..\dataset\Flickr8k_text\Flickr8k.token.txt'
    # load captions
    doc = load_doc(filename)
    # parse captions
    captions = load_captions(doc)
    print(f'Loaded captions for {len(captions)} images')
    # clean captions
    clean_captions(captions)
    # summarize vocabulary
    vocabulary = create_vocabulary(captions)
    print(f'Vocabulary size: {len(vocabulary)}')
    # save to file
    save_captions(captions, '../resources/captions.txt')
