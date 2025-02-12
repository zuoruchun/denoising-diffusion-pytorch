class BaseModel:
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        print("BaseModel forward")
        return x

class MyModel(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("MyModel forward") 
        return x * 2

class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, data):
        result = self.model(data)
        print(result)

# 创建实例
my_model = MyModel()
trainer = Trainer(my_model)
trainer.train(5)

base_model = BaseModel()
trainer1 = Trainer(base_model)
trainer1.train(5)