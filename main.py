# from lib.dataImage import DataImage
# from experimentation.testVitesse import launch
from scripts.record.record_race import Record
from scripts.preprocess_data.preprocess import Dataset
from scripts.drive.drive_car import Drive

dataset_folder = 'train_races_mid'

def main():
    print("\n\nWelcome to AI Trackmania !\n\n")
    print("What do you want to do ?")
    print("1 : Record your races.")
    print("2 : Preprocess my dataset.")
    print("3 : Drive at full speed !")
    valide_choice = False
    while not valide_choice:
        try:
            choice = int(input("Enter your choice :"))
            valide_choice = choice == 1 or choice == 2 or choice == 3
        except:
            valide_choice = False
        if not(valide_choice) : 
            print("It was not a valid number! Try again !")

    if choice == 1:
        # Init record object
        r = Record(dataset_folder)
        # Record race
        r.record()
        # Save the record.
        r.saveRecord()
    if choice == 2:
        # Init dataset object
        dataset = Dataset(dataset_folder)
        # Preprocess the data
        dataset.processingData()
        # Saving it
        dataset.saveDataset()
    if choice == 3:
        d = Drive("model2.h5",True)
        d.start()

if __name__ == "__main__":
    main()