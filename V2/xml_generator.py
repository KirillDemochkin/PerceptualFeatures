import xml.etree.ElementTree as ET
import random

INPUT_FPATH = "./imei.txt"
OUTPUT_FPATH = "./generated.xml"

tree = ET.parse('whitelist.xml')
root = tree.getroot()


def make_xml(imei):
    for elem in root.iter('Header'):
        elem.set('TransactionId',
                 'Harman_Device_{:06d}{:06d}'.format(random.randint(0, 999999), random.randint(0, int(10e5 - 1))))

    for elem in root.iter('Device'):
        elem.set('IMEI', imei)

    for elem in root.iter('Account'):
        elem.set('MSISDN', '{:06d}{:05d}'.format(random.randint(0, int(10e5 - 1)), random.randint(0, int(10e4 - 1))))

    for elem in root.iter('SIM'):
        elem.set('IMSI', '{:06d}{:05d}{:04d}'.format(random.randint(0, int(10e5 - 1)), random.randint(0, int(10e4 - 1)),
                                                     random.randint(0, int(10e3 - 1))))

    for elem in root.iter('SIM'):
        elem.set('ICCID',
                 '{:08d}{:08d}{:04d}'.format(random.randint(0, int(10e7 - 1)), random.randint(0, int(10e7 - 1)),
                                             random.randint(0, int(10e3 - 1))))


def main():
    with open(INPUT_FPATH, 'r') as fin:
        lines = fin.readlines()

    lines = [l.split(',') for l in lines]
    xmls = [make_xml(str.strip(l[0])) for l in lines]
    with open(OUTPUT_FPATH, 'w+') as fout:
        ET.dump(fout)


if __name__ == '__main__':
    main()
