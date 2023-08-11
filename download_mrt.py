import requests

base_url = 'http://archive.routeviews.org/bgpdata/2012.01/UPDATES/'

start_day = 1
end_day = 31
start_hour = 0
end_hour = 23
start_minute = 0
end_minute = 45

for day in range(start_day, end_day + 1):
    day_str = f'{day:02d}'
    
    for hour in range(start_hour, end_hour + 1):
        hour_str = f'{hour:02d}'
        
        for minute in range(start_minute, end_minute + 1, 15):
            minute_str = f'{minute:02d}'
            
            url = f'{base_url}updates.201201{day_str}.{hour_str}{minute_str}.bz2'
            
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(f'updates.201201{day_str}.{hour_str}{minute_str}.bz2', 'wb') as f:
                    f.write(response.content)
                print(f'Downloaded: {url}')
            else:
                print(f'Failed to download: {url}')
