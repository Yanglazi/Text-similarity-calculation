def f():
    with open('blogs/blogs.train.txt','r',encoding='utf-8') as f:
        for line in f:
            yield line

for i in f():
    print(i)
