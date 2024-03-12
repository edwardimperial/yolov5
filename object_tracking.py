##import relevant libraries
import pandas as pd
from io import StringIO  # Required to create a file-like object from a string
import math
import cv2
import os


# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)  #'warn' , 'raise' or None




##read text files 
import pandas
##columns = ["class","x", "y", "h", "w"]
df = pd.DataFrame()  ##creat empty dataframe for appending later
#df_single = pd.DataFrame()



### test code (Edward)


###

##read text files 
for index in range(335, 1000):  # Replace  with the number of files you want to open
    front_portion = f'adcs_3_{index}'
    directory_path = "object_tracking_RawData/Camera3_labels/"
    matching_files = [file for file in os.listdir(directory_path) if file.startswith(front_portion)][0]
    filename = os.path.join(directory_path, matching_files)
    
    try:
        with open(filename, 'r') as file:
            file_content = file.read()
            data = pd.read_csv(StringIO(file_content), sep=' ', header =None) ##convert the string into dataframe
            data.columns = ["class","centriod_x", "centriod_y", "h", "w", "conf_level"]
            #data.columns = ["class","centriod_x", "centriod_y", "h", "w"]
            #columns_to_drop = ['class','h','w',"conf_level"]
            data = data.drop(columns = 'class', axis = 1)
            data["image_number"] = index  ##add in the index of text file opened (leds are in which images)
            data["led_assignment"] = 0
            data["file_name"] = filename
            df = pandas.concat([df, data], ignore_index=True) ##append new data into df
    except FileNotFoundError:
        print(f'{filename} not found.')

        

##automatic initialisation
df["led_assignment"] = 0
first_img_number = df['image_number'][0]
index = df.index[df['image_number'] == first_img_number].tolist()
led = df[0:len(index)].sort_values(by='centriod_x')
for i in range(len(index)):
    df['led_assignment'][i] = led.index[i] + 1

##extract the time from the file name
df['time'] = 0
for i in range(len(df)):
    df["time"][i] = df['file_name'][i][52:-4]
    
df = df.drop(columns = 'file_name', axis = 1)

## Consolidated code

def object_tracking():
    new_assignment = max(df['led_assignment'][0:3]) ##initialise new led assignment

    for img_number in range(335,999):  ##based on the image number
        index = df.index[df["image_number"] == img_number].tolist()    ##the indices of the current image
        index_next = df.index[df["image_number"] == img_number+1].tolist()  ##the indices of the next image
        if img_number > 0:
            index_pre = df.index[df["image_number"] == img_number-1].tolist()  ##the indices of the previous image
        for i in index:                                  ##for each index in the current image
            euclidean_dist_list = []
            j_list = []
            for j in index_next:                         ##for each index in the next image
                j_list.append(j)
                euclidean_dist = math.sqrt((df["centriod_x"][i]-df["centriod_x"][j])**2 + (df["centriod_y"][i]-df["centriod_y"][j])**2)
                euclidean_dist_list.append(euclidean_dist)
            table = pd.DataFrame({'index_j': j_list, 'euclidean_dist': euclidean_dist_list})
            value_index = table.loc[table['euclidean_dist'] == min(table['euclidean_dist'])].index.tolist()
            number_index = table['index_j'][value_index[0]]
            df["led_assignment"][number_index] = df["led_assignment"][i]
        
        if len(index_next) > len(index):              ##increasing in the number of led
            if len(index) >= len(index_pre):          ##
                if not df.loc[(df['led_assignment'] == 0 ) & (df['image_number'] == img_number+1)].empty:
                    index_new_point = df.loc[(df['led_assignment'] == 0 )& (df['image_number'] == img_number+1)].index.tolist()
                    #print("index_nex_point", format(index_new_point))
                    df_no_assignment = df.iloc[index_new_point[0]]
                    comparison_value = math.sqrt((df_no_assignment["centriod_x"]-step_down_data_info["centriod_x"])**2 + (df_no_assignment["centriod_y"]-step_down_data_info["centriod_y"])**2)
                    #print(step_down_data_info)
                    #print("comparison value = ",format(comparison_value))  
                    if comparison_value <= 0.1:     ##can adjust the treshold value        ##sudden appearence of old led
                        led_assignment = step_down_data_info['led_assignment']
                        #print(led_assignment)
                        df["led_assignment"][index_new_point[0]] = led_assignment
                    
                    else:             ##new led 
                        index_new_point = df.loc[(df['led_assignment'] == 0 )& (df['image_number'] == img_number+1)].index.tolist()
                        new_pt_index = df.loc[df['image_number'] == img_number].index.tolist() ##find the index of the image_number
                        new_assignment = new_assignment + 1
                        if new_assignment == 5:
                            new_assignment = 1
                        else:
                            pass
                        df["led_assignment"][index_new_point[0]] = new_assignment
                        print(f"New LED detected: {new_assignment} in image: {img_number}")
                else:
                    pass
            elif len(index) < len(index_pre): ##check for the potential mislabellings
                led_list_pre = []
                led_list = []
                for i in index_pre:
                    led = df['led_assignment'][i]
                    led_list_pre.append(led)
                for i in index:
                    led = df['led_assignment'][i]
                    led_list.append(led)
                # Extract values in index_pre that are not in index (the mislabeled led)
                values_not_in_index_pre = [value for value in led_list_pre if value not in led_list]
                #print(values_not_in_index_pre)
                if len(values_not_in_index_pre) == 1:
                    print(f"Potential mislabeling in image: {img_number} - LED: {values_not_in_index_pre[0]}, please verify")
                    if not df.loc[(df['led_assignment'] == 0 ) & (df['image_number'] == img_number+1)].empty:
                        index_new_point = df.loc[(df['led_assignment'] == 0 )& (df['image_number'] == img_number+1)].index.tolist()
                        new_assignment_check = values_not_in_index_pre[0]
                        df["led_assignment"][index_new_point[0]] = new_assignment_check
                elif len(values_not_in_index_pre) > 1:
                    print(f"Potential mislabeling in image: {img_number} - LED: {values_not_in_index_pre}, please verify")
                    for i in index_pre:
                        euclidean_dist_list = []
                        j_list = []
                        for j in index_next:
                            j_list.append(j)
                            euclidean_dist = math.sqrt((df["centriod_x"][i]-df["centriod_x"][j])**2 + (df["centriod_y"][i]-df["centriod_y"][j])**2)
                            euclidean_dist_list.append(euclidean_dist)
                        table = pd.DataFrame({'index_j': j_list, 'euclidean_dist': euclidean_dist_list})
                        #print(table)
                        value_index = table.loc[table['euclidean_dist'] == min(table['euclidean_dist'])].index.tolist()
                        number_index = table['index_j'][value_index[0]]
                        df["led_assignment"][number_index] = df["led_assignment"][i]
                

        elif len(index_next) < len(index):  ## if the the number of led decreases
            for i in index_next:
                df["led_assignment"][i] = 0
            for i in index_next:
                euclidean_dist_list = []
                j_list = []
                led_assign_list = []
                for j in index:
                    j_list.append(j)
                    euclidean_dist = math.sqrt((df["centriod_x"][i]-df["centriod_x"][j])**2 + (df["centriod_y"][i]-df["centriod_y"][j])**2)
                    euclidean_dist_list.append(euclidean_dist)
                    led_assign = df['led_assignment'][j]
                    led_assign_list.append(led_assign)
                table = pd.DataFrame({'index_j': j_list, 'euclidean_dist': euclidean_dist_list, 'led_assign':led_assign_list })
                value_index = table.loc[table['euclidean_dist'] == min(table['euclidean_dist'])].index.tolist()
                led_assign = table['led_assign'][value_index[0]]
                df["led_assignment"][i] = led_assign
            current_led_list = []
            for j in index_next:    ##find and store the information of the led that is disappearing
                current_led = df["led_assignment"][j]
                current_led_list.append(current_led)
            #print(img_number)
            #print("data step down")
            #print(led_assign_list)
            #print(current_led_list)
            #Extract values in index_pre that are not in index (stepped down led)
            values_not_in_index_pre_1 = [value for value in led_assign_list if value not in current_led_list]
            #print("values_not_in_index_pre_1 = {}".format(values_not_in_index_pre_1) )
            step_down_data_info = df.loc[(df['led_assignment'] == values_not_in_index_pre_1[0] ) & (df['image_number'] == img_number)]
            columns_to_drop = ['h','w','conf_level']
            step_down_data_info = step_down_data_info.drop(columns = columns_to_drop, axis = 1)
            #print(step_down_data_info)
                
        
        else:
            pass
            
             
            
    return df

object_tracking_res = object_tracking()   

#print(object_tracking_res)

             