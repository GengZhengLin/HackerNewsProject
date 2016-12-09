import csv
import numpy as np
from collections import defaultdict

csv_filepath = "data.csv"

time_points_list_dict = defaultdict(list)
all_points_list = []
with open(csv_filepath,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    field_names=reader.fieldnames
    for row in reader:
        num_points = int(row['num_points'])
        created_at = row['created_at']
        hour = int(created_at.split()[1].split(':')[0])
        time_points_list_dict[hour].append(num_points)

output_file = "created_hour_average_points.csv"
time_news_num_list = []
with open(output_file, 'w') as csv_output:
    csv_output.write('created_hour,average_num_points,30_percentile_num_points,'+
                     '60_percentile_num_points,80_percentile_num_points,90_percentile_num_points\n')
    for hour, points_list in time_points_list_dict.items():
        csv_output.write(str(hour)+','+str(np.mean(points_list).round(3))+','
                         +str(np.percentile(points_list,30))+','
                         +str(np.percentile(points_list,60))+','
                         +str(np.percentile(points_list,80))+','
                         +str(np.percentile(points_list,90))+'\n')
        time_news_num_list.append(len(points_list))






