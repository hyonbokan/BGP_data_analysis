import pybgpstream
import pprint

stream = pybgpstream.BGPStream()

stream.add_filter('record-type', 'updates')
stream.add_filter('project', 'routeviews')
stream.set_data_interface('singlefile')
stream.set_data_interface_option('singlefile', 'rib-file', 'updates.20230102.0230')

for elem in stream:
    pprint.pprint(elem.fields)

print("Test Finished")