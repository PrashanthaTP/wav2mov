lines = ['aytala\n','macha']
with open('test.txt','a+') as f:
    f.writelines(lines)
    
with open('test.txt','r') as f: 
    read_lines = f.read().split('\n')
    print(read_lines)
    

