class LicensePlateMetric():

    def __init__(self,length_plate_number):
        self.length_plate_number = length_plate_number
        self.accumulation_number_correct_characters = [0]*(length_plate_number+1)
        self.number_groundtruth = 0

    def update(self,
               plates_groundtruth,
               plates_prediction,
               matched,
               unmatched_groundtruth,
               unmatched_prediction):

        #self.number_groundtruth += len(matched)+len(unmatched_groundtruth)
        for i,j in matched:
            plate_groundtruth = plates_groundtruth[i]
            plate_prediction  = plates_prediction[j]
            if "?" in plate_groundtruth:
                continue
            number_correct_characters = sum([plate_groundtruth[k] == plate_prediction[k]\
                                            for k in range(self.length_plate_number)])
            self.accumulation_number_correct_characters[ number_correct_characters ] += 1
            self.number_groundtruth += 1

        for i in unmatched_groundtruth:
            plate_groundtruth = plates_groundtruth[i]
            if "?" in plate_groundtruth:
                continue
            self.number_groundtruth += 1

        #for j in unmatched_prediction:
        #    plate_prediction = plates_prediction[j]
        self.accumulation_number_correct_characters[0] += len(unmatched_prediction)

    def get_accuracy(self):
        spacing = "- "*27
        print(spacing + "\nLicense Plate Accuracy\n" + spacing)
        for number_correct_characters in range(self.length_plate_number,4,-1):
            print
            print("criterion: %d character(s) are matched %s"%\
                (number_correct_characters,\
                 " (all matched)" if (number_correct_characters == self.length_plate_number) else ""))
            TP = sum(self.accumulation_number_correct_characters[number_correct_characters:])
            FP = sum(self.accumulation_number_correct_characters[:number_correct_characters])
            FN = self.number_groundtruth-TP
            print("TP = %d, FP = %d, FN = %d"%(TP,FP,FN))
            print("precision: %6.2f %%"%(1e2*TP/(TP+FP)))
            print("   recall: %6.2f %%"%(1e2*TP/self.number_groundtruth))
        print(spacing)


# TODO add a example?
#if __name__ == "__main__":
#    import numpy as np
#
#    LENGTH_PLATE_NUMBER = 7
#    NUMBER_FRAMES = 10
#
#    def license_plate_generator(number_frames=10,p_plate=0.5,length_plate_number=7):
#        license_plates = []
#        for no_frame in range(number_frames):
#            if np.random.rand() > 1-p_plate:
#                license_plates.append(["%0*d"%(length_plate_number,np.random.randint(10**length_plate_number))])
#            else:
#                license_plates.append([])
#        return license_plates
#
#    plates_groundtruth = license_plate_generator(number_frames=NUMBER_FRAMES)
#    plates_prediction  = license_plate_generator(number_frames=NUMBER_FRAMES)
#    lp_metric = LicensePlateMetric(length_plate_number=LENGTH_PLATE_NUMBER)
#    for i in range(NUMBER_FRAMES):
#        lp_metric.update(plates_groundtruth[i],plates_prediction[i])
#    lp_metric.get_results()
