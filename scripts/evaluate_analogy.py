# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to evaluate given word embeddings on the word analogy task.
"""
import argparse
from numpy import array
from numpy import dot
from numpy import linalg

def read_normalized_embeddings(embedding_path, no_counts, ignore_line1):
    """Reads normalized embeddings."""
    embedding = {}
    dim = 0
    line1_ignored = False
    with open(embedding_path, "r") as embedding_file:
        for line in embedding_file:
            tokens = line.split()
            if len(tokens) > 0:
                if ignore_line1 and (not line1_ignored):
                    line1_ignored = True
                    continue
                if no_counts: # <word> <values...>
                    word = tokens[0]
                    starting_index = 1
                else: # <count> <word> <values...>
                    word = tokens[1]
                    starting_index = 2
                values = []
                for i in range(starting_index, len(tokens)):
                    values.append(float(tokens[i]))
                if dim:
                    assert(len(values) == dim)
                else:
                    dim = len(values)
                embedding[word] = array(values)
                embedding[word] /= linalg.norm(embedding[word])  # Make norm 1.
    return embedding, dim

def read_analogy_data(analogy_path):
    """Reads analogy questions."""
    questions = {} # analogy_type -> [questions...]
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

    return questions, num_questions, vocab

def assign_embedding(embedding, vocab):
    """
    Assigns an embedding for each word in the vocabulary, if any.
    """
    assigned_embedding = {}
    for word in vocab:
        if word in embedding:
            # If we have the exact match in embeddings, just use it.
            assigned_embedding[word] = embedding[word]
        elif word.lower() in embedding:
            # If we do *not* have the exact match but a lowercased match in
            # embeddings, we will use that instead: e.g., "Microsoft" in
            # assigned_embedding might refers to "microsoft" in embedding.
            # This can happen when the corpus was lowercased.
            assigned_embedding[word] = embedding[word.lower()]

    return assigned_embedding

def predict(w1, w2, v1, embedding, method):
    """
    Finds x in vocab V such that w1:w2 ~ v1:x using embeddings.
    """
    # Get a relevant embedding for each word type {w1, w2, v1}.
    w1_embedding = embedding[w1]
    w2_embedding = embedding[w2]
    v1_embedding = embedding[v1]
    assert(len(w1_embedding) > 0 and len(w2_embedding) > 0 and
           len(v1_embedding) > 0)

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
        print("Unknown method: {0}".format(method))
        exit(0)

    predicted_v2 = None
    max_score = float("-inf")
    for word in embedding:
        if (word == w1) or (word == w2) or (word == v1):
            # Do not consider the same words that appear in the question.
            continue
        v_embedding = embedding[word]
        assert len(v_embedding) > 0
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

def main(args):
    """
    Each line is an analogy question containing 5 items t w1 w2 v1 v2:
       - Analogy type t (e.g., family)
       - 4 words w1, w2, v1, v2 (e.g., groom bride king queen)
    The task is to correctly predict v2 using w1, w2, v1.
    """
    embedding, dim = read_normalized_embeddings(args.embedding_path,
                                                args.no_counts,
                                                args.ignore_line1)
    print("{0} {1}-dimensional embeddings.".format(len(embedding), dim))

    questions, num_questions, vocab = read_analogy_data(args.analogy_path)
    print("{0} analogy questions ({1} categories)".format(num_questions,
                                                          len(questions)))

    assigned_embedding = assign_embedding(embedding, vocab)
    print("{0}/{1} word types have embeddings.".format(len(assigned_embedding),
                                                       len(vocab)))

    num_answered = 0
    num_correct = 0
    num_answered_pertype = {}
    num_correct_pertype = {}
    correct_answers = {}
    wrong_answers = {}
    for analogy_type in questions:
        for (w1, w2, v1, v2) in questions[analogy_type]:
            if (w1 in assigned_embedding) and (w2 in assigned_embedding) and \
                   (v1 in assigned_embedding) and (v2 in assigned_embedding):
                num_answered += 1
                if not analogy_type in num_answered_pertype:
                    num_answered_pertype[analogy_type] = 0
                    num_correct_pertype[analogy_type] = 0
                num_answered_pertype[analogy_type] += 1
                predicted_v2 = predict(w1, w2, v1, assigned_embedding,
                                       args.method)
                if predicted_v2 == v2:
                    num_correct += 1
                    num_correct_pertype[analogy_type] += 1
                    if not analogy_type in correct_answers:
                        correct_answers[analogy_type] = []
                    correct_answers[analogy_type].append((w1, w2, v1, v2))
                else:
                    if not analogy_type in wrong_answers:
                        wrong_answers[analogy_type] = []
                    wrong_answers[analogy_type].append((w1, w2, v1, v2,
                                                        predicted_v2))

    acc = float(num_correct) / num_answered * 100
    print("Acc: {0:.2f} ({1}/{2})".format(acc, num_correct, num_answered))
    for analogy_type in num_answered_pertype:
        acc_pertype = float(num_correct_pertype[analogy_type]) / \
                      num_answered_pertype[analogy_type] * 100
        print("\t{0:.2f} -- {1} ({2}/{3})".format(
            acc_pertype, analogy_type, num_correct_pertype[analogy_type],
            num_answered_pertype[analogy_type]))

    if args.report:
        with open(args.report, "w") as report_file:
            for analogy_type in correct_answers:
                report_file.write("\n{0}/{1} correct in {2}\n".format(
                    num_correct_pertype[analogy_type],
                    num_answered_pertype[analogy_type], analogy_type))
                for (w1, w2, v1, v2) in correct_answers[analogy_type]:
                    report_file.write("\t{0} : {1} ~ {2} : {3}\n".format(
                        w1, w2, v1, v2))
            report_file.write("\n\n")
            for analogy_type in wrong_answers:
                report_file.write("\n{0}/{1} wrong in {2}\n".format(
                    num_answered_pertype[analogy_type] -
                    num_correct_pertype[analogy_type],
                    num_answered_pertype[analogy_type], analogy_type))
                for (w1, w2, v1, v2, predicted_v2) in \
                    wrong_answers[analogy_type]:
                    report_file.write(
                        "\t{0} : {1} ~ {2} : {3} \t (predicted {4})\n".format(
                            w1, w2, v1, v2, predicted_v2))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("analogy_path", type=str, help="path to word "
                           "analogy file")
    argparser.add_argument("embedding_path", type=str, help="path to word "
                           "embeddings file")
    argparser.add_argument("--method", type=str, default="mult",
                           help="Inference method (default: %(default)d)")
    argparser.add_argument("--report", type=str, help="path to detailed report")
    argparser.add_argument("--no_counts", action="store_true", help="embeddings"
                           " don't counts for the first column?")
    argparser.add_argument("--ignore_line1", action="store_true", help="ignore "
                           "the first line in the embeddings file?")
    parsed_args = argparser.parse_args()
    main(parsed_args)
