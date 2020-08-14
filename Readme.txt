This is the description file that supports the supplementary codes for Vein-ID.

This code operates in two phase.

1- Vein - Extraction:

    * Go to folder 'Vein_Extraction'.
    * This folder has two *.m script files labelled as 'img_v.m' and 'setpblock'(both of these scripts work togather to perform vein extraction).
    * Additionally, this folder has a sub-folder titles as 'Data' which contains the original IR and Depth images (captured with Intel D415) 
      for a user to demostrate the extraction capabilities of proposed solutions.
    * Note that, data of only one user is included here as the size of whole data set is big to be accomodated.
    * To see the extracted vein, go to 'img_v.m' and provide the complete path of provided data (appropriate position in the code is indicated).
    * Run the code.
    * Code will automatically read all the images one by one and will write the extracted vein pattern in a folder named 'Up'.

2. Identification Code:

    * Go to folder 'Idn'
    * This folder has two *.m files labelled as 'Classf_code_cnn.m' and 'Classf.code_autoenc.m'.
    * Additionally, this folder has a sub-folder that contains the extrcted vein-patterns of 5 users as example (Note that: These pattern are extracted leveraging the same method 
       as described above. For verification, you can compare the vein-pattern of U10 include in this folder with vein-pattern extracted on you local machine by employing the method 
       described above).
    * To evaluate the performance of both CNN and Stacked-Autoencoder, go to corresponding '*.m' file and provide the complete path of processed vein-patterns (i.e., the provided patterns).
    * Run the code to view the results.
    * You can change the parameters like training samples to see its impact.


  NOTE: These codes have been tested on MATLAB 2017b, 2018a, and 2019a. Results included in this folder are on R2019a with no GPU.


    *********+++++++++++++++++***************+++++++++++++++++*****************++++++++++++++****************************
                      Syed. W. Shah (on behalf of co-authors), UNSW, 2020.



