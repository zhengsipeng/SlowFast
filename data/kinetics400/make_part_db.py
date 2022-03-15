import os
import sys
import tqdm
import csv
import pdb

blob_dir = "/home/v-sizheng/blob/teamdrive/kinetics400_cvdf"
def make_part_db(split):
    csvFile = open("%s_clean.csv"%split)
    reader = csv.reader(csvFile)    
    
    new_csv = []
    for line in reader:
        new_csv.append(line)
        if reader.line_num == 1:
            continue
        vid, start, end = line[1: 4]
        vname = "%s_%06d_%06d.mp4"%(vid, int(start), int(end))
        os.system("cp %s/%s/%s %s"%(blob_dir, split, vname, split))
        #pdb.set_trace()
        if reader.line_num > 100:
            break
    csvFile.close()

    csvFile = open("%s_part.csv"%split, "w")
    writer = csv.writer(csvFile)
    for line in new_csv:
        writer.writerow(line)
    csvFile.close()


def make_kinetics_db(split):
    csvFile = open("%s_part.csv"%split)
    reader = csv.reader(csvFile)
    new_csv = []
    for line in reader:
        if reader.line_num == 1:
            continue
        new_csv.append(line)
    csvFile.close()

    video_dir = "/home/v-sizheng/Desktop/SlowFast/data/kinetics400/"
    csvFile = open("%s.csv"%split, "w") 
    writer = csv.writer(csvFile, delimiter=" ")
    
    for line in new_csv:
        vid, start, end, clsid = line[1], int(line[2]), int(line[3]), int(line[-1])
        videopath = "%s/%s/%s_%06d_%06d.mp4"%(video_dir, split, vid, start, end)
        writer.writerow([videopath, clsid])
    csvFile.close()
   

if __name__ == "__main__":
    if sys.argv[1] == "mv_part_db":
        make_part_db(sys.argv[2])
    if sys.argv[1] == "make_kinetics_db":
        make_kinetics_db(sys.argv[2])