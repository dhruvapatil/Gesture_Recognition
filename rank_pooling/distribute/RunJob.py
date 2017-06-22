#from Machines import machines
import subprocess
import time
from distribute.Misc import get_machine_count


'''
#code to divide the task of sliding window exhaustive feature selection across
#five cross_validation data to distribute over 160 machines
scale_list = [1,5]
for i in range(160):
    fold_index = i/32
    scale = scale_list[(i / 16) % 2]
    list_index = i%16
    print machines[i], fold_index, scale, list_index
    #subprocess.call(["run.sh",machines[i],str(fold_index), str(scale), str(list_index)])
'''

'''
#code to divide rank pooling feature selection for 12 features across
#five cross validation data distributed over 160 machines
for i in range(160):
    fold_index = i/32
    list_index = i%32
    print i, machines[i],fold_index, list_index
    #subprocess.call(["run.sh",machines[i],str(fold_index), str(list_index)])
    #time.sleep(0.5)
'''


machines = ['albany', 'atlanta', 'annapolis', 'augusta', 'austin', 'baton-rouge', 'bismarck', 'boise', 'boston', 'carson-city', 'charleston', 'columbia', 'columbus-oh', 'concord', 'denver', 'des-moines', 'dover', 'frankfort', 'harrisburg', 'hartford', 'helena', 'honolulu', 'indianapolis', 'jackson', 'jefferson-city', 'juneau', 'lansing', 'lincoln', 'little-rock', 'madison', 'montgomery', 'montpelier', 'nashville', 'oklahoma-city', 'olympia', 'phoenix', 'pierre', 'providence', 'raleigh', 'richmond', 'sacramento', 'saint-paul', 'salem', 'salt-lake-city', 'santa-fe', 'springfield', 'tallahassee', 'topeka', 'trenton', 'a-basin', 'ajax', 'beaver-creek', 'breckenridge', 'buttermilk', 'cooper', 'copper-mtn', 'crested-butte', 'eldora', 'grandy-ranch', 'aspen-highlands', 'howelsen-hill', 'keystone', 'loveland', 'mary-jane', 'monarch', 'powderhorn', 'purgatory', 'silverton', 'snowmass', 'steamboat', 'sunlight', 'telluride', 'vail', 'winter-park', 'wolf-creek', 'earth', 'jupiter', 'mars', 'mercury', 'neptune', 'saturn', 'uranus', 'venus', 'bentley', 'bugatti', 'ferrari', 'jaguar', 'lamborghini', 'lotus', 'maserati', 'porsche', 'corvette', 'mustang', 'ankara', 'baghdad', 'bangkok', 'beijing', 'berlin', 'bogota', 'cairo', 'damascus', 'dhaka', 'hanoi', 'hong-kong', 'jakarta', 'kabul', 'kinshasa', 'lima', 'london', 'madrid', 'mexico-city', 'moscow', 'pyongyang', 'riyadh', 'santiago', 'seoul', 'singapore', 'tehran', 'tokyo', 'washington-dc', 'almond', 'beech', 'brazil', 'butternut', 'cashew', 'chestnut', 'chinquapin', 'coconut', 'filbert', 'ginko', 'hazelnut', 'heartnut', 'hickory', 'lychee', 'macadamia', 'nangai', 'peanut', 'pecan', 'pepita', 'pili', 'pinion', 'shea', 'walnut', 'coconuts', 'raspberries', 'pomegranates', 'dates', 'eggplant', 'endive', 'fennel', 'garlic', 'gourd', 'horseradish', 'kale', 'kelp', 'leek', 'lettuce', 'mushroom', 'okra', 'onion', 'parsley', 'parsnip', 'pea', 'pepper', 'potato', 'pumpkin', 'radish', 'rhubarb', 'romanesco', 'rutabaga', 'shallot', 'spinach', 'squash', 'tomatillo', 'tomato', 'turnip', 'wasabi', 'yam', 'zucchini', 'figs', 'grapes', 'huckleberries', 'kiwis', 'melons', 'nectarines', 'peaches', 'pears', 'kitchener', 'acushla', 'twins-tower', 'oak', 'damavand', 'bacon', 'eggs', 'chip', 'copper', 'kenai', 'kuskokwim', 'stikine', 'buzz', 'crackle', 'sun', 'beaverhead', 'cut-bank', 'depuy', 'gallatin', 'jagger', 'lion', 'jefferson', 'gardner', 'platte', 'ruby', 'stillwater', 'boulder', 'red-rock', 'shields', 'betaz', 'eolus', 'kaiso', 'laramie', 'st-vrain', 'strauss', 'yampa', 'k2', 'makalu', 'athabasca', 'recondite', 'alexandra', 'bryce', 'clemenceau', 'diadem', 'fryatt', 'tsar', 'la-plata', 'frankfurt', 'somerset', 'animas', 'arkansas', 'colorado', 'conejos', 'dolores', 'eagle', 'frying-pan', 'gunnison', 'piedra', 'rio-grande', 'bighorn', 'clarks-fork', 'green', 'greybull', 'hams-fork', 'new-fork', 'nowood', 'popo-agie', 'powder', 'shoshone', 'snake', 'sweetwater', 'tongue', 'wind', 'yellowstone', 'lang', 'pancakes', 'waffles', 'toast', 'blue', 'cyan', 'magenta', 'pink', 'yellow', 'temple', 'apples', 'oranges', 'europa', 'sherman', 'faure', 'poudre', 'volga', 'massive', 'halloween', 'blanca', 'humboldt', 'huron', 'bross']

machine_count = get_machine_count()
for i in range(machine_count):
    list_index = i
    print machines[i], list_index
    subprocess.call(["run.sh",machines[i],str(list_index)])
    time.sleep(0.5)

