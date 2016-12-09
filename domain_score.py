import csv
import numpy as np
# from urlparse import urlparse
from collections import defaultdict
import tldextract

csv_filepath = "data.csv"

domain_points_list_dict = defaultdict(list)
all_points_list = []
with open(csv_filepath,'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    field_names=reader.fieldnames
    for row in reader:
        num_points = int(row['num_points'])
        all_points_list.append(num_points)
        url = row['url']
        # parsed_uri = urlparse(url)
        # domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
        domain = tldextract.extract(url).domain
        domain_points_list_dict[domain].append(num_points)

print 'domain number:', len(domain_points_list_dict)
print 'average points:', np.mean(all_points_list)

print np.percentile(all_points_list, 10)
print np.percentile(all_points_list, 20)
print np.percentile(all_points_list, 30)
print np.percentile(all_points_list, 40)
print np.percentile(all_points_list, 50)
print np.percentile(all_points_list, 60)
print np.percentile(all_points_list, 70)
print np.percentile(all_points_list, 80)
print np.percentile(all_points_list, 90)

# output_file = "domain_average_points.csv"
# domain_news_num_list = []
# with open(output_file, 'w') as csv_output:
#     csv_output.write('domain,average_num_points\n')
#     for domain, points_list in domain_points_list_dict.items():
#         csv_output.write(domain+','+str(np.mean(points_list))+'\n')
#         domain_news_num_list.append(len(points_list))
#
# print np.mean(domain_news_num_list)
# print np.percentile(domain_news_num_list, 10)
# print np.percentile(domain_news_num_list, 20)
# print np.percentile(domain_news_num_list, 30)
# print np.percentile(domain_news_num_list, 40)
# print np.percentile(domain_news_num_list, 50)
# print np.percentile(domain_news_num_list, 60)
# print np.percentile(domain_news_num_list, 70)
# print np.percentile(domain_news_num_list, 80)
# print np.percentile(domain_news_num_list, 90)





