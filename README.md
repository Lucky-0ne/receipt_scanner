# receipt_scanner
A repo for my combined project ideas from the Deep Learning and Learning from Images courses to reduce the stress of clearing the housing fund receipts.

## Initial Idea
I wanted to simplify the accounting process in my shared flat. When my flatmate and I go shopping, we keep the bills and after a while we manually go through them and check who paid what. We end up with three categories (flatmate A, flatmate B & common amount). The process so far has been for me to manually enter each amount into Excel to calculate the final amount for each category to see who gets money back.  
  
My idea was to use a combination of color recognition and OCR to reduce the hassle of manually typing in each amount and instead automatically scan and calculate the amounts. (The categorization would still have to be done manually as there is no recurring pattern behind which cash value on the receipt belongs to which category).  
  
  
For this idea I wanted to programm an App that uses the live video stream of a webcam to detect color coded cash values, scan the regions of interest (ROIs) and predicts the displayed float value with the help of an OCR model. After validating the predictions, which are displayed alongside the live video stream, you would confirm the scan and save the values to a .csv file, for example. When the final cash receipt is scanned, the predictions are confirmed and the results saved, a simple (e.g. sum) operation would return the final amount of each category.  
  
Initially I wanted to use an off-the-shelf model from huggingface, kaggle or similar sites, but after some experimentation I found that there was no really functional model available for my approach. I also wanted to gain some hands-on experience training an OCR model on my own using my self-generated labelled image data. Due to the ongoing cybersecurity issue at my university, the cluster is still out of service and I wasn't able to pursue this approach any further for the time being.

## Current Approach
That's the reason why I came up with an alternate approach for the moment. I'm still using the app to detect color coded regions of interest (ROIs), but instead of predicting the cash amounts during the live video stream (due to the lack of an off-the-shelf or self-trained model), I will instead store the [unlabelled image snippets](https://github.com/Lucky-0ne/receipt_scanner/tree/main/main/images/result_snippets) as .png and then label them using Google's Vision AI. This model is free to use for < 1000 API requests per month. Because of this restriction I added a [preprocessing step](https://github.com/Lucky-0ne/receipt_scanner/blob/main/main/scripts/classify_images.ipynb#L5) to concatenate the image snippets into a single large canvas to reduce the number of API calls from a few hundred to one.  
  
I then save the [labelled image snippets](https://github.com/Lucky-0ne/receipt_scanner/tree/main/main/images/classified_snippets) to manually validate its accuracy. For my first test run of > 500 image snippets the Google Vision AI only failed in two cases, of which I would dismiss [one](https://github.com/Lucky-0ne/receipt_scanner/blob/main/main/images/result_snippets/orange/discarded/2024-03-27_10-56-16_orange_ROI_5.png) as negligible because of its (very!) poor image quality. The [other one](https://github.com/Lucky-0ne/receipt_scanner/blob/main/main/images/result_snippets/orange/discarded/2024-03-27_12-01-09_orange_ROI_10.png) provides material for further [discussion]().  
  
After validation, I simply read all the filenames, convert them back to floats and calculate the final amounts. Even though I would like to avoid the steps of manually validating large amounts of labelled image data in my final approach by implementing a live prediction during the live video stream, I have to say that I'm positively surprised at how easy and simple the scanning, detection and storage process is, and at how exceptionally well the Google Vision AI model performs for a basically free to use approach.

## Future Plans
As mentioned previously, I would like to implement a live prediction with a (self-trained) local model and further improve the GUI. For example, adding a slider for the color detection mode to calibrate the detected color spectrum should be done with ease. Furthermore I did some tests to detect/extract the corresponding items that belong to each cash value (e.g. tomatoes, milk, etc.), but since I lost some progress due to the hack and since I have no interest in this information at the moment, I discarded the idea for now.

## Challenges
Aside from the many challenges I've already overcome in the course of this project, there's one major problem that I'm currently only able to avoid by manual debugging, and that's only tolerable at the moment because the Vision AI model works with a near 100% success rate. When concatenating the image snippets into a large canvas to reduce the number of API calls, I abandon the explicit mapping of image to label. For now, I only check at the end whether the Google prediction after [postprocessing]() returns as many labels as the original number of image snippets, and if so, I do another check by manually checking each labelled image snippet. If the first assertion ever fails (which happens almost 0% of the time), I have to manually check which image snippet caused the failed prediction, exclude it from the process, and rerun the prediction. This could be prevented by not plotting all image snippets on a large canvas, i.e. predicting each snippet individually and thus finding the problematic snippets faster, but this would very quickly exceed the limit of < 1000 API calls per month. But because of the very low error rate AND because the whole thing with Google Vision AI is only a temporary solution, I refrain from a more in-depth search for a solution.  
  
Another big challenge would be whether it's even possible to train a model on these labelled image data. As I mentioned in my project presentation, it may be necessary to simplify the approach by iterating through each digit of the float value and recreating the whole value at the end, but due to the setback caused by the hack, I'll have to look into this at a later time.
