from advina import adock

score = adock('./proteins/6rqu.pdbqt', 'O=C([O-])CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]', 'test02')

print(score)
