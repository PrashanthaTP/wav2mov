from typing import List


class Log:
    def __init__(self,log_type):
        self.type = log_type
        self.vals = {}
        
    
    def add(self,key,value):
        self.vals[key] = value 
        
    def items(self):
        return self.vals.items()

class LogsTracker: 
    def __init__(self,logs:List[Log]):
        self.logs = {}
        for log in logs: 
            self.logs[log.type] = log 
            
    def get_type(self,log_type):
        return self.logs.get(log_type,None)
        
 