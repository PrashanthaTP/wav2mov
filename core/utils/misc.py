from colorama import init,Fore,Back,Style
from .os_utils import is_windows
from tqdm import tqdm 

init() # colorama


class COLOURS:
    RED = Fore.RED
    GREEN = Fore.GREEN
    BLUE = Fore.BLUE 
    YELLOW = Fore.YELLOW 
    WHITE = Fore.WHITE

# def colored(text,color=None,reset=True):
#     if color is None:
#         color = COLOURS.WHITE
#     return f"{color}{text}{Style.RESET_ALL}" if reset else  f"{color}{text}"

def __coloured(text:str,colour)->str:
    return f"{colour}{text}{Style.RESET_ALL}"

def error(text:str)->str:
    return  __coloured(text,COLOURS.RED)

def debug(text:str)->str:
    return __coloured(text,COLOURS.BLUE)

def success(text:str)->str:
    return __coloured(text,COLOURS.GREEN)





def string_wrapper(msg):
    if is_windows():
        return debug(msg)
    else:
        return msg
    
def get_tqdm_iterator(iterable,description,colour="green"):
    """setups the tqdm progress bar

    Arguments:
        `iterable` {Iterable} 
        `description` {str} -- Description to be shown in progress bar

    Keyword Arguments:
        `colour` {str} -- colour of progress bar (default: {"green"})

    Returns:
        `tqdm` 
    """
    return tqdm(iterable,ascii=True,desc=description,total=len(iterable),colour=colour)

