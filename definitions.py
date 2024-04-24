##############################################################
###################### NAMESPACE #############################
##############################################################

MDF_RENAMER = {
    '@input.Affinity': 'EMOTIONAL', 
    '@input.MeetNeeds': 'RATIONAL', 
    '@input.Dynamic': 'LEADERSHIP', 
    '@input.Unique': 'UNIQUENESS',
    'Meaningful': 'GRATIFICATION', 
    'Different': 'DISTINCTION', 
    'Salient': 'AMPLIFICATION',
    'Power': 'POWER', 
    'Premium v2': 'PREMIUM'}


FACTORS = ['EMOTIONAL', 'RATIONAL', 'LEADERSHIP', 'UNIQUENESS']
MDFS = ['GRATIFICATION', 'DISTINCTION', 'AMPLIFICATION']
POWER_PREMIUM = ['POWER', 'PREMIUM']

MDFS_POWER_PREMIUM = ['GRATIFICATION', 'DISTINCTION', 'AMPLIFICATION', 'POWER', 'PREMIUM']


# edge keys
FROM_KEY, TO_KEY, TYPE_KEY  = 'from', 'to', 'type'
EDGE_KEYS = [FROM_KEY, TO_KEY, TYPE_KEY]

MDF_GRAPH = [
    {'from': 'EMOTIONAL', 'to': 'GRATIFICATION', 'type': 'mdf'},
    {'from': 'RATIONAL', 'to': 'GRATIFICATION', 'type': 'mdf'},
    {'from': 'LEADERSHIP', 'to': 'GRATIFICATION', 'type': 'mdf'},
    {'from': 'UNIQUENESS', 'to': 'GRATIFICATION', 'type': 'mdf'},
    {'from': 'EMOTIONAL', 'to': 'DISTINCTION', 'type': 'mdf'},
    {'from': 'RATIONAL', 'to': 'DISTINCTION', 'type': 'mdf'},
    {'from': 'LEADERSHIP', 'to': 'DISTINCTION', 'type': 'mdf'},
    {'from': 'UNIQUENESS', 'to': 'DISTINCTION', 'type': 'mdf'},
    {'from': 'EMOTIONAL', 'to': 'AMPLIFICATION', 'type': 'mdf'},
    {'from': 'RATIONAL', 'to': 'AMPLIFICATION', 'type': 'mdf'},
    {'from': 'LEADERSHIP', 'to': 'AMPLIFICATION', 'type': 'mdf'},
    {'from': 'UNIQUENESS', 'to': 'AMPLIFICATION', 'type': 'mdf'},
    {'from': 'GRATIFICATION', 'to': 'POWER', 'type': 'mdf'},
    {'from': 'DISTINCTION', 'to': 'POWER', 'type': 'mdf'},
    {'from': 'AMPLIFICATION', 'to': 'POWER', 'type': 'mdf'},
    {'from': 'GRATIFICATION', 'to': 'PREMIUM', 'type': 'mdf'},
    {'from': 'DISTINCTION', 'to': 'PREMIUM', 'type': 'mdf'},
    {'from': 'AMPLIFICATION', 'to': 'PREMIUM', 'type': 'mdf'}
]

# edge types
EDGE_TYPE_MDF = 'mdf'
EDGE_TYPE_PATH = 'path'
EDGE_TYPE_CORR = 'correlation'
EDGE_TYPES = ['path', 'correlation', 'mdf']


SOE_COL_TEMPLATE = 'soe.IMG{:02d}_'