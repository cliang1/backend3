class ExerciseModel:
    _instance = None
    
    def __init__(self):
        self.model = None
        self.dt = None
        self.features = ['id', 'diet', 'time', 'kind']
        self.target = 'pulse'
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.exercise_data = []

    @staticmethod
    def get_instance():   
        if ExerciseModel._instance is None:
            ExerciseModel._instance = ExerciseModel()
        return ExerciseModel._instance

    def init_exercise_list(self):
        self._init_exercise_data()

    def _init_exercise_data(self):
        exercise_list = [
            {"id": 0, "diet": "low fat", "pulse": 85, "time": "1 min", "kind": "rest"},
            {"id": 1, "diet": "low fat", "pulse": 85, "time": "15 min", "kind": "rest"},
            {"id": 2, "diet": "low fat", "pulse": 88, "time": "30 min", "kind": "rest"},
            {"id": 3, "diet": "low fat", "pulse": 90, "time": "1 min", "kind": "rest"},
            {"id": 4, "diet": "low fat", "pulse": 92, "time": "15 min", "kind": "rest"},
            {"id": 5, "diet": "low fat", "pulse": 93, "time": "30 min", "kind": "rest"},
            {"id": 6, "diet": "low fat", "pulse": 97, "time": "1 min", "kind": "rest"},
            {"id": 7, "diet": "low fat", "pulse": 97, "time": "15 min", "kind": "rest"},
            {"id": 8, "diet": "low fat", "pulse": 94, "time": "30 min", "kind": "rest"},
            {"id": 9, "diet": "low fat", "pulse": 80, "time": "1 min", "kind": "rest"},
            {"id": 10, "diet": "low fat", "pulse": 82, "time": "15 min", "kind": "rest"},
            {"id": 11, "diet": "low fat", "pulse": 83, "time": "30 min", "kind": "rest"},
            {"id": 12, "diet": "low fat", "pulse": 91, "time": "1 min", "kind": "rest"},
            {"id": 13, "diet": "low fat", "pulse": 92, "time": "15 min", "kind": "rest"},
            {"id": 14, "diet": "low fat", "pulse": 91, "time": "30 min", "kind": "rest"},
            {"id": 15, "diet": "no fat", "pulse": 83, "time": "1 min", "kind": "rest"},
            {"id": 16, "diet": "no fat", "pulse": 83, "time": "15 min", "kind": "rest"},
            {"id": 17, "diet": "no fat", "pulse": 84, "time": "30 min", "kind": "rest"},
            {"id": 18, "diet": "no fat", "pulse": 87, "time": "1 min", "kind": "rest"},
            {"id": 19, "diet": "no fat", "pulse": 88, "time": "15 min", "kind": "rest"},
            {"id": 20, "diet": "no fat", "pulse": 90, "time": "30 min", "kind": "rest"},
            {"id": 21, "diet": "no fat", "pulse": 92, "time": "1 min", "kind": "rest"},
            {"id": 22, "diet": "no fat", "pulse": 94, "time": "15 min", "kind": "rest"},
            {"id": 23, "diet": "no fat", "pulse": 95, "time": "30 min", "kind": "rest"},
            {"id": 24, "diet": "no fat", "pulse": 97, "time": "1 min", "kind": "rest"},
            {"id": 25, "diet": "no fat", "pulse": 99, "time": "15 min", "kind": "rest"},
            {"id": 26, "diet": "no fat", "pulse": 96, "time": "30 min", "kind": "rest"},
            {"id": 27, "diet": "no fat", "pulse": 100, "time": "1 min", "kind": "rest"},
            {"id": 28, "diet": "no fat", "pulse": 97, "time": "15 min", "kind": "rest"},
            {"id": 29, "diet": "no fat", "pulse": 100, "time": "30 min", "kind": "rest"},
            {"id": 30, "diet": "low fat", "pulse": 86, "time": "1 min", "kind": "walking"},
            {"id": 31, "diet": "low fat", "pulse": 86, "time": "15 min", "kind": "walking"},
            {"id": 32, "diet": "low fat", "pulse": 84, "time": "30 min", "kind": "walking"},
            {"id": 33, "diet": "low fat", "pulse": 93, "time": "1 min", "kind": "walking"},
            {"id": 34, "diet": "low fat", "pulse": 103, "time": "15 min", "kind": "walking"},
            {"id": 35, "diet": "low fat", "pulse": 104, "time": "30 min", "kind": "walking"},
            {"id": 36, "diet": "low fat", "pulse": 90, "time": "1 min", "kind": "walking"},
            {"id": 37, "diet": "low fat", "pulse": 92, "time": "15 min", "kind": "walking"},
            {"id": 38, "diet": "low fat", "pulse": 93, "time": "30 min", "kind": "walking"},
            {"id": 39, "diet": "low fat", "pulse": 95, "time": "1 min", "kind": "walking"},
            {"id": 40, "diet": "low fat", "pulse": 96, "time": "15 min", "kind": "walking"},
            {"id": 41, "diet": "low fat", "pulse": 100, "time": "30 min", "kind": "walking"},
            {"id": 42, "diet": "low fat", "pulse": 89, "time": "1 min", "kind": "walking"},
            {"id": 43, "diet": "low fat", "pulse": 96, "time": "15 min", "kind": "walking"},
            {"id": 44, "diet": "low fat", "pulse": 95, "time": "30 min", "kind": "walking"},
            {"id": 45, "diet": "no fat", "pulse": 84, "time": "1 min", "kind": "walking"},
            {"id": 46, "diet": "no fat", "pulse": 86, "time": "15 min", "kind": "walking"},
            {"id": 47, "diet": "no fat", "pulse": 89, "time": "30 min", "kind": "walking"},
            {"id": 48, "diet": "no fat", "pulse": 103, "time": "1 min", "kind": "walking"},
            {"id": 49, "diet": "no fat", "pulse": 109, "time": "15 min", "kind": "walking"},
            {"id": 50, "diet": "no fat", "pulse": 90, "time": "30 min", "kind": "walking"},
            {"id": 51, "diet": "no fat", "pulse": 92, "time": "1 min", "kind": "walking"},
            {"id": 52, "diet": "no fat", "pulse": 96, "time": "15 min", "kind": "walking"},
            {"id": 53, "diet": "no fat", "pulse": 101, "time": "30 min", "kind": "walking"},
            {"id": 54, "diet": "no fat", "pulse": 97, "time": "1 min", "kind": "walking"},
            {"id": 55, "diet": "no fat", "pulse": 98, "time": "15 min", "kind": "walking"},
            {"id": 56, "diet": "no fat", "pulse": 100, "time": "30 min", "kind": "walking"},
            {"id": 57, "diet": "no fat", "pulse": 102, "time": "1 min", "kind": "walking"},
            {"id": 58, "diet": "no fat", "pulse": 104, "time": "15 min", "kind": "walking"},
            {"id": 59, "diet": "no fat", "pulse": 103, "time": "30 min", "kind": "walking"},
            {"id": 60, "diet": "low fat", "pulse": 93, "time": "1 min", "kind": "running"},
            {"id": 61, "diet": "low fat", "pulse": 98, "time": "15 min", "kind": "running"},
            {"id": 62, "diet": "low fat", "pulse": 110, "time": "30 min", "kind": "running"},
            {"id": 63, "diet": "low fat", "pulse": 98, "time": "1 min", "kind": "running"},
            {"id": 64, "diet": "low fat", "pulse": 104, "time": "15 min", "kind": "running"},
            {"id": 65, "diet": "low fat", "pulse": 112, "time": "30 min", "kind": "running"},
            {"id": 66, "diet": "low fat", "pulse": 98, "time": "1 min", "kind": "running"},
            {"id": 67, "diet": "low fat", "pulse": 105, "time": "15 min", "kind": "running"},
            {"id": 68, "diet": "low fat", "pulse": 99, "time": "30 min", "kind": "running"},
            {"id": 69, "diet": "low fat", "pulse": 87, "time": "1 min", "kind": "running"},
            {"id": 70, "diet": "low fat", "pulse": 132, "time": "15 min", "kind": "running"},
            {"id": 71, "diet": "low fat", "pulse": 120, "time": "30 min", "kind": "running"},
            {"id": 72, "diet": "low fat", "pulse": 94, "time": "1 min", "kind": "running"},
            {"id": 73, "diet": "low fat", "pulse": 110, "time": "15 min", "kind": "running"},
            {"id": 74, "diet": "low fat", "pulse": 116, "time": "30 min", "kind": "running"},
            {"id": 75, "diet": "no fat", "pulse": 95, "time": "1 min", "kind": "running"},
            {"id": 76, "diet": "no fat", "pulse": 126, "time": "15 min", "kind": "running"},
            {"id": 77, "diet": "no fat", "pulse": 143, "time": "30 min", "kind": "running"},
            {"id": 78, "diet": "no fat", "pulse": 100, "time": "1 min", "kind": "running"},
            {"id": 79, "diet": "no fat", "pulse": 126, "time": "15 min", "kind": "running"},
            {"id": 80, "diet": "no fat", "pulse": 140, "time": "30 min", "kind": "running"},
            {"id": 81, "diet": "no fat", "pulse": 103, "time": "1 min", "kind": "running"},
            {"id": 82, "diet": "no fat", "pulse": 124, "time": "15 min", "kind": "running"},
            {"id": 83, "diet": "no fat", "pulse": 140, "time": "30 min", "kind": "running"},
            {"id": 84, "diet": "no fat", "pulse": 94, "time": "1 min", "kind": "running"},
            {"id": 85, "diet": "no fat", "pulse": 135, "time": "15 min", "kind": "running"},
            {"id": 86, "diet": "no fat", "pulse": 130, "time": "30 min", "kind": "running"},
            {"id": 87, "diet": "no fat", "pulse": 99, "time": "1 min", "kind": "running"},
            {"id": 88, "diet": "no fat", "pulse": 111, "time": "15 min", "kind": "running"},
            {"id": 89, "diet": "no fat", "pulse": 150, "time": "30 min", "kind": "running"}
        ]
        self.exercise_data = exercise_list

    def _clean(self):
        for exercise in self.exercise_data:
            exercise['time'] = int(exercise['time'].split()[0])
        self.exercise_data = pd.DataFrame(self.exercise_data)
        self.exercise_data.drop(['id', 'diet'], axis=1, inplace=True)
        onehot = self.encoder.fit_transform(self.exercise_data[['kind']]).toarray()
        cols = ['kind_' + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols)
        self.exercise_data = pd.concat([self.exercise_data, onehot_df], axis=1)
        self.features.extend(cols)
        self.exercise_data.drop(['kind'], axis=1, inplace=True)
        self.exercise_data.dropna(inplace=True)

    def _train(self):
        X = self.exercise_data[self.features]
        y = self.exercise_data[self.target]
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)

    def predict_pulse(self, person):
        person_df = pd.DataFrame(person, index=[0])
        pulse = np.squeeze(self.model.predict_proba(person_df))
        return {'pulse': pulse}
