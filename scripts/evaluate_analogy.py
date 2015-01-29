# Author: Karl Stratos (karlstratos@gmail.com)
"""
This module is used to evaluate given word embeddings on the word analogy task.

Argument 1 [input 1: each line = analogy_type w1 w2 v1 v2]
Argument 2 [input 2: word embeddings file]
"""
import sys
from math import sqrt
from numpy import array
from numpy import dot
from numpy import linalg

def read_normalized_embeddings(embedding_path):
    """Read normalized embeddings."""
    embedding = {}
    dim = 0
    with open(embedding_path, "r") as embedding_file:
        for line in embedding_file:
            tokens = line.split()
            if len(tokens) > 0:
                word = tokens[1]
                values = []
                for i in range(2, len(tokens)):
                    values.append(float(tokens[i]))
                if dim:
                    assert len(values) == dim
                else:
                    dim = len(values)
                embedding[word] = array(values)
                embedding[word] /= linalg.norm(embedding[word])  # Make norm 1.
    return embedding, dim

def predict(w1, w2, v1, embedding, method):
    """
    Finds x in vocab V such that w1:w2 ~ v1:x using embeddings.
    """
    # Get a relevant embedding for each word type {w1, w2, v1}.
    w1_embedding = embedding[w1]
    w2_embedding = embedding[w2]
    v1_embedding = embedding[v1]
    assert(len(w1_embedding) > 0 and len(w2_embedding) > 0
           and len(v1_embedding) > 0)

    if method == "add":
        # argmax_{x in V \ {w1, w2, v1}} cos(x, w2 - w1 + v1)
        target_embedding = w2_embedding - w1_embedding + v1_embedding
        target_embedding /= linalg.norm(target_embedding)
    elif method == "add-simple":
        # argmax_{x in V \ {w1, w2, v1}} cos(x, w2 + v1)
        target_embedding = w2_embedding  + v1_embedding
        target_embedding /= linalg.norm(target_embedding)
    elif method == "raw":
        # argmax_{x in V \ {w1, w2, v1}} cos(x, v1)
        target_embedding = v1_embedding
    elif method == "mult" or method == "mult-simple":
        # argmax_{x in V \ {w1, w2, v1}}
        #    scos(x, v1) * scos(c, w2) / (scos(x, w1) + 0.001)
        # where scos(x, y) := (cos(x, y) + 1) / 2
        pass
    else:
        print "Unknown method: {0}".format(method)
        exit(0)

    predicted_v2 = None
    max_score = float("-inf")
    for word in embedding:
        if (word == w1) or (word == w2) or (word == v1):
            # Do not consider the same words that appear in the question.
            continue
        v_embedding = embedding[word]
        assert(len(v_embedding) > 0)
        if method == "mult":
            # Compute cosine similarities and put them in [0,1].
            cos_v1 = (dot(v1_embedding, v_embedding) + 1) / 2
            cos_w2 = (dot(w2_embedding, v_embedding) + 1) / 2
            cos_w1 = (dot(w1_embedding, v_embedding) + 1) / 2

            # Multiply these shifted cosine similarities.
            score = cos_v1 * cos_w2 / (cos_w1 + 0.001)
        elif method == "mult-simple":
            # Compute cosine similarities and put them in [0,1].
            cos_v1 = (dot(v1_embedding, v_embedding) + 1) / 2
            cos_w2 = (dot(w2_embedding, v_embedding) + 1) / 2

            # Multiply these shifted cosine similarities.
            score = cos_v1 * cos_w2
        else:
            # Compute cosine similarity with the target embedding.
            score = dot(target_embedding, v_embedding)
        if score > max_score:
            max_score = score
            predicted_v2 = word
    return predicted_v2

def evaluate_analogy(analogy_path, embedding_path, method):
    """
    Each line is an analogy question containing 5 items t w1 w2 v1 v2:
       - Analogy type t (e.g., family)
       - 4 words w1, w2, v1, v2 (e.g., groom bride king queen)
    The task is to correctly predict v2 using w1, w2, v1.
    """
    # Read normalized embeddings.
    embedding, dim = read_normalized_embeddings(embedding_path)
    print "Read {0} {1}-dimensional embeddings".format(len(embedding), dim)

    # Read in analogy questions.
    questions = {}
    num_questions = 0
    vocab = {}
    with open(analogy_path, "r") as analogy_file:
        for line in analogy_file:
            tokens = line.split()
            if len(tokens) > 0:
                analogy_type = tokens[0]
                w1 = tokens[1]
                w2 = tokens[2]
                v1 = tokens[3]
                v2 = tokens[4]
                vocab[w1] = True
                vocab[w2] = True
                vocab[v1] = True
                vocab[v2] = True
                if not analogy_type in questions:
                    questions[analogy_type] = []
                questions[analogy_type].append((w1, w2, v1, v2))
                num_questions += 1
    print "Read {0} analogy questions ({1} categories)".format(num_questions,
                                                               len(questions))

    embedding2 = {}
    for word in vocab:
        if word in embedding:
            embedding2[word] = embedding[word]
        elif word.lower() in embedding:
            embedding2[word] = embedding[word.lower()]
    print "{0} out of {1} word types have corresponding embeddings".format(
        len(embedding2), len(vocab))

    # Make predictions.
    num_answered = 0
    num_correct = 0
    num_skipped = 0
    num_answered_pertype = {}
    num_correct_pertype = {}
    for analogy_type in questions:
        for (w1, w2, v1, v2) in questions[analogy_type]:
            if (w1 in embedding2) and (w2 in embedding2) and \
                   (v1 in embedding2) and (v2 in embedding2):
                num_answered += 1
                if not analogy_type in num_answered_pertype:
                    num_answered_pertype[analogy_type] = 0
                    num_correct_pertype[analogy_type] = 0
                num_answered_pertype[analogy_type] += 1
                predicted_v2 = predict(w1, w2, v1, embedding2, method)
                if predicted_v2 == v2:
                    num_correct += 1
                    num_correct_pertype[analogy_type] += 1
                    print "(RIGHT) {0} : {1} == {2} : {3}".format(
                        w1, w2, v1, predicted_v2, v2)
                else:
                    print "(WRONG) {0} : {1} == {2} : {3} (answer: {4})".format(
                        w1, w2, v1, predicted_v2, v2)
            else:
                num_skipped += 1
    print "\nSkipped {0} questions that lack embeddings".format(num_skipped)

    acc = float(num_correct) / num_answered * 100
    print "Acc: {0:.2f} ({1}/{2}) -- all".format(acc, num_correct, num_answered)
    for analogy_type in num_answered_pertype:
        acc_pertype = float(num_correct_pertype[analogy_type]) / \
                      num_answered_pertype[analogy_type] * 100
        print "\tAcc: {0:.2f} ({1}/{2}) -- {3}".format(
            acc_pertype, num_correct_pertype[analogy_type],
            num_answered_pertype[analogy_type], analogy_type)

if __name__ == "__main__":
    # Path to word analogy file.
    ANALOGY_PATH = sys.argv[1]

    # Path to word embeddings file.
    EMBEDDING_PATH = sys.argv[2]

    # Prediction method.
    METHOD = sys.argv[3]

    evaluate_analogy(ANALOGY_PATH, EMBEDDING_PATH, METHOD)
