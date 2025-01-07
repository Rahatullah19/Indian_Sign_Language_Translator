def wer(reference, hypothesis):
    """
    Compute the Word Error Rate (WER) between a reference sentence and a hypothesis sentence.
    """
    r = reference.split()
    h = hypothesis.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]

    for i in range(1, len(r) + 1):
        d[i][0] = i
    for j in range(1, len(h) + 1):
        d[0][j] = j

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / len(r)
