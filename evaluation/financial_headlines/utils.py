import datasets
import numpy as np
import sklearn


label_map = { "negative": "negative", "neutral": "neutral", "positive": "positive"}


def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = ["negative", "neutral", "positive"]

        doc["label"] = label_map[doc["Price Sentiment"]]
        
        doc["sentence"] = doc["News"]

        return doc
    
    filtered_dataset = dataset.filter(lambda x: x["Price Sentiment"] != "none")
    

    return filtered_dataset.map(_helper)


def f1(predictions, references):
    return (predictions[0], references[0])


def agg_f1_weighted(items):
    predictions, references = zip(*items)
    references, predictions = np.asarray(references), np.asarray(predictions)

    fscore = sklearn.metrics.f1_score(references, predictions, average="weighted")
    return fscore

def accuracy(predictions, references):
    return (predictions[0], references[0])


def agg_accuracy(items):
    predictions, references = zip(*items)
    references, predictions = np.asarray(references), np.asarray(predictions)

    fscore = sklearn.metrics.accuracy(references, predictions)
    return fscore