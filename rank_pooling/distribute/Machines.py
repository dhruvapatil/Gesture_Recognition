import paramiko



def get_available_machines():
    machine_list = open("/s/bach/i/sys/info/machines").readlines()[2:]
    machines = [m.split("\t")[0] for m in machine_list if "Linux(Fedora)" in m and "lattice" not in m]
    print machines
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    count = 0
    connected_machines = []
    for m in machines:
        try:
            ssh.connect(m,timeout=1)
            print count, m
            count += 1
            connected_machines.append(m)
        except:
            print "Cant connect to ", m

    print 'Active machines are: ', count
    print connected_machines

#get_available_machines()
machines = ['albany', 'annapolis', 'atlanta', 'augusta', 'austin', 'baton-rouge', 'bismarck', 'boise', 'boston', 'carson-city', 'charleston', 'columbia', 'columbus-oh', 'concord', 'denver', 'des-moines', 'dover', 'frankfort', 'harrisburg', 'hartford', 'helena', 'honolulu', 'indianapolis', 'jackson', 'jefferson-city', 'juneau', 'lansing', 'lincoln', 'little-rock', 'madison', 'montgomery', 'montpelier', 'nashville', 'oklahoma-city', 'olympia', 'phoenix', 'pierre', 'providence', 'raleigh', 'richmond', 'sacramento', 'saint-paul', 'salem', 'salt-lake-city', 'santa-fe', 'springfield', 'tallahassee', 'topeka', 'trenton', 'a-basin', 'ajax', 'beaver-creek', 'breckenridge', 'buttermilk', 'cooper', 'copper-mtn', 'crested-butte', 'eldora', 'grandy-ranch', 'aspen-highlands', 'howelsen-hill', 'keystone', 'loveland', 'mary-jane', 'monarch', 'powderhorn', 'purgatory', 'silverton', 'snowmass', 'steamboat', 'sunlight', 'telluride', 'vail', 'winter-park', 'wolf-creek', 'earth', 'jupiter', 'mars', 'mercury', 'neptune', 'saturn', 'uranus', 'venus', 'bentley', 'bugatti', 'ferrari', 'jaguar', 'lamborghini', 'lotus', 'maserati', 'porsche', 'corvette', 'mustang', 'ankara', 'baghdad', 'bangkok', 'beijing', 'berlin', 'bogota', 'cairo', 'damascus', 'dhaka', 'hanoi', 'hong-kong', 'jakarta', 'kabul', 'kinshasa', 'lima', 'london', 'madrid', 'mexico-city', 'moscow', 'pyongyang', 'riyadh', 'santiago', 'seoul', 'singapore', 'tehran', 'tokyo', 'washington-dc', 'almond', 'beech', 'brazil', 'butternut', 'cashew', 'chestnut', 'chinquapin', 'coconut', 'filbert', 'ginko', 'hazelnut', 'heartnut', 'hickory', 'lychee', 'macadamia', 'nangai', 'peanut', 'pecan', 'pepita', 'pili', 'pinion', 'shea', 'walnut', 'coconuts', 'raspberries', 'pomegranates', 'dates', 'eggplant', 'endive', 'fennel', 'garlic', 'gourd', 'horseradish', 'kale', 'kelp', 'leek', 'lettuce', 'mushroom', 'okra', 'onion', 'parsley', 'parsnip', 'pea', 'pepper', 'potato', 'pumpkin', 'radish', 'rhubarb', 'romanesco', 'rutabaga', 'shallot', 'spinach', 'squash', 'tomatillo', 'tomato', 'turnip', 'wasabi', 'yam', 'zucchini', 'figs', 'grapes', 'huckleberries', 'kiwis', 'melons', 'nectarines', 'peaches', 'pears', 'kitchener', 'acushla', 'twins-tower', 'oak', 'damavand', 'bacon', 'eggs', 'chip', 'copper', 'kenai', 'kuskokwim', 'stikine', 'buzz', 'crackle', 'sun', 'beaverhead', 'cut-bank', 'depuy', 'gallatin', 'jagger', 'lion', 'jefferson', 'gardner', 'platte', 'ruby', 'stillwater', 'boulder', 'red-rock', 'shields', 'betaz', 'eolus', 'kaiso', 'laramie', 'st-vrain', 'strauss', 'yampa', 'k2', 'makalu', 'athabasca', 'recondite', 'alexandra', 'bryce', 'clemenceau', 'diadem', 'fryatt', 'tsar', 'la-plata', 'frankfurt', 'somerset', 'animas', 'arkansas', 'colorado', 'conejos', 'dolores', 'eagle', 'frying-pan', 'gunnison', 'piedra', 'rio-grande', 'bighorn', 'clarks-fork', 'green', 'greybull', 'hams-fork', 'new-fork', 'nowood', 'popo-agie', 'powder', 'shoshone', 'snake', 'sweetwater', 'tongue', 'wind', 'yellowstone', 'lang', 'pancakes', 'waffles', 'toast', 'blue', 'cyan', 'magenta', 'pink', 'yellow', 'temple', 'apples', 'oranges', 'europa', 'sherman', 'faure', 'poudre', 'volga', 'massive', 'halloween', 'blanca', 'humboldt', 'huron', 'bross']
