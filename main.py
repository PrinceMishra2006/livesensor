from sensor.exception import SensorException
import os 
import sys 

def test_exception():
    try:
        a=1/0
    except Exception as e:
        raise SensorException(e,sys)




if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        print(e)