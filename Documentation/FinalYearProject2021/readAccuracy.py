import json, ast

object = open(r"../Results/val_history.json","r+")
val = ast.literal_eval(object.read())
acc = 0

for key in val.keys():
    acc = acc + val[key]['val_accuracy'][0]

avg_acc = acc / len(val.keys())

print(avg_acc)