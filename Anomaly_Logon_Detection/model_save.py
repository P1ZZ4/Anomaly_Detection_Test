k = np.arange(1,20)
acc =[]
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    acc.append(knn.score(X_test,y_test))

for i,j in zip(k,acc):
    print('k:' + str(i)+', acc = ', str(j))
    

knn = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)

## model save (pkl file
import pickle
model = open("system.pkl",'wb')
pickle.dump(knn,model)
loaded_model = pickle.load(open('system.pkl', 'rb'))
result = loaded_model.predict(X_test)


