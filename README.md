Abstract

This work studies the use of supervised learning to build an efficient genre classifier, which
predicts the genre of a musical piece among Jazz, Country, Classical and Disco. Machine learning
paved the way for further advances in the field of genre classification, and its automation finds
many applications in music services relying on large databases. The first challenge in building the
classifier is the extraction of relevant features from the available dataset. In this paper, several
basic spectral and timbre descriptors are examined and compared, such as chroma features and
Mel Frequency Cepstral Coefficients (MFCC).
The second challenge, concerns the configuration of the Support Vector Machine classifier. Several
parameters tuning the model’s complexity are determined following a k-fold cross-validation. The
configured model is finally trained on 300 pieces and tested on 100 pieces with a precision score of
0.89, and the resulting confusion matrix reflects the musical affinities between jazz and classical
music, and disco and country music.


Introduction

The aim with this project was to train a model via machine learning, so that it would be able to
differentiate between four different music genres i.e. Country, Classical, Disco and Jazz.
The data set that was used for this task was the GTZAN Genre Collection (http://marsyas.info/
downloads/datasets.html). This data set included 100 songs from the different genres, for a total
of 400 tracks considered. Each song has a duration of 30 seconds, a sampling rate of 22050 Hz and
is in ”.au” format. All songs in the data set were already divided into genres by folder, this was
fundamental to implement our supervised learning algorithm.
This paper describes the main steps needed to implement our music genre classification system, plus
some insights about the code structure and our conclusions about the overall performances.

The work was divided into the following main steps:
 Preprocessing
 Feature extraction
 Feature selection
 Hyperparameters optimization
 Training and cross-validation


Preprocessing

Before extracting all the features from the dataset, a preprocessing step is performed.

Approach
Each song is loaded from a local path, and thus converted to a time sequence of float
values. The sequence is normalized, in order to be represented with values in [−1.0, 1.0]. The features
we chose to use depend on different representations of the song; all these models are evaluated from
the time series in the preprocessing phase.

Representations

Windowed Audio
It is a matrix that contains the windowed version of the original time sequence;
each column is a windowed frame. This model is particularly useful to evaluate temporal features in
a shorter time interval, and allows to estimate an averaged behaviour.

Short Time Fourier Transform
It is the time-frequency representation of the song. It is used
to calculate spectral features for every frame (they are then averaged), or to estimate the evolution
of the song’s spectrum.

Chroma Spectrum
It is the chromagram of the song, a time-frequency representation that expresses the harmonic and melodic evolution of a musical piece.
Mel Frequency Cepstrum It is a spectral model based on the cosine transform of the Mel power
spectrum. This representation mimics the auditory system’s response, and its use in music genre
classification problems is widespread.

Mel Frequency Cepstrum
It is a spectral model based on the cosine transform of the Mel power
spectrum. This representation mimics the auditory system’s response, and its use in music genre
classification problems is widespread.


Features
After the preprocessing of the data is done, the subsequent step is the computation and the extraction
of the features.

Approach
First, a preliminary analysis on the choice of features was made. Although is known in
literature which features are more relevant for a genre classification problem, the chosen approach
was to gather as many low-level descriptors as possible, in order to collect the largest possible pool of
information from the signals. This choice is justified by the necessity of also compare the performances
of the different parameters and different combinations of selected features.
Of course, this means that a further mechanism of features selection is implemented later, to define
which features are relevant for the classification problem in object.
In addition, a rough prediction on which features represent a fine descriptor for the classification can
be made, based on their graphical output.

Feature listing
The features that were computed are enlisted and explained below. In particular,
for each descriptor, the feature corresponds to its mean value and, when specified, also to its standard
deviation.

 Zero Crossing Rate
It is computed for every frame x of windowed audio as
****
For the zero crossing rate, the standard deviation is evaluated alongside the mean value.

 Spectral Centroid
It is computed from the time-frequency matrix X as
****
By visualizing the curves of the spectral centroid mean, we can consider it as a good feature to
consider in order to distinguish the ’disco’ class.

 Spectral Flux
It measures how quickly the power spectrum varies from frame by frame. It is computed as
****

 Spectral Spread
It describes the average deviation of the spectrum around its centroid, and is computed as
****

 Spectral Roll-Off
It is the frequency below a certain specified percentage of the toatl spectral energy lies. The
percentage chosen for the problem in object was 85%. The frequency is evaluated as
****

 Spectral Decrease
The spectral decrease aims at quantifying the amount of decrease of the amplitude spectrum.
Coming from perceptive studies, it is supposed to be more correlated with human perception.
It quantifies the amount of decrease of the amplitude spectrum, and it is more correlated with
human perception. It is computed as
****

 Mel-Frequency Cepstral Coefficients (MFCCs)
The Mel-Frequency Coefficient are extracted as described in the previous section. In particular,
for every coefficients from 2 to 14 the mean value and the standard deviation were evaluated.
Those features are among the most powerful for genre classification. As expected, it was possible
to highlight some relevant features, based on their graphics. In particular, by looking at the
curves generated from the mean values and the standard deviations of the first coefficients, it
was possible to distinguish reasonably different shapes for different classes.

On the other hand, the curves of the last 5 coefficients’ standard deviation show some real
clustering between disco and country on one side, and between classical and jazz on the other.
This last aspect is justified if we consider the type of instrument that are used for those genres:
more rhythmic based instruments for the former, more classical and less rhythmic instruments
for the latter.

 Chroma Features
As described in the previous chapter, the chromagram was computed in the pre-processing stage
and was used to calculate different features, many of which are already described above, as the
Chroma Centroid, Chroma Spread, Chroma Flux. In particular, their mean value was taken
into account. In addition to these, the mean value and the standard definition for each chroma
coefficient were extracted, as well as the chroma bins associated to the maximum and minimum
energy. The performances will be extensively discussed later, but it was clear from the graphics
that they tended to perform quite nicely.

 Onset Events
This feature is a descriptor of how many onset events, or roughly the transient signals, occur in
a specific amount of time. Specifically, it was chosen to count the total number of ”hits” over
the whole duration of the song, obtaining a normalization of ”hits per second”.
The choice of using also a rhythmic feature to a genre classification problem was done to
evaluate if the presence of some recurring patterns in the different genres could be considered as
a classifier. Anyway, the results of this feature alone were not satisfying enough. Nonetheless,
it is a welcome addition even for a modest increase in the accuracy of the system.


Features Selection
As already specified before, not every feature contributes significantly to the
classification problem. The filtering was made using a variance threshold selector. The reasoning
behind the choice lies in the fact that the higher the variance of a feature, the more spread are the
curves. Consequently a class, or a cluster of classes is easier to recognize with respect to the specific
descriptor.
Therefore, only the features that showed a variance higher than a set threshold value were considered
relevant for the test. However it is also important to specify that the threshold parameter is subject
to further fine-tuning in the successive steps.

MFCC and Chroma Features
A final consideration can be made on the two most used features
used in music genre classification. Those are both implemented in the system, and even though their
performances are reasonably comparable, the MFCC features still seem to generally perform slightly
better with respect to the dataset used, as shown in Figure 4.


Training and cross-validation
Now that we have a clearer view of our features, we can start to build our classifier. Our strategy is
to train an SVM-classifier, and to optimize some of its parameters using cross-validation.
Splitting the data First, we split the dataset into a training set of 300 samples and a testing
set of 100 samples. Later, the training set will be divided into k=10 subsets of 30 samples for
cross-validation.

Setting the SVM classifier
We set an SVM-classifier with a one-versus-one decision function.
This type of classifier creates binary classifiers for each pair of class and then uses majority voting.
It is often used for its good performances in multiclass-classification problems.
The “hyperparameters” that we will optimize during the cross-validation are:

 kernel: the rbf kernel is the default kernel. It provides good performances, but a linear or
polynomial kernel might also be very effective with the right degree;
 poly degree: in case of polynomial kernel;
 regularization: to adjust the complexity of the model, and how well it has to fit to the training
data;
 min var: minimum variance threshold for the features’ selection.

Parameters optimization
For each configuration, i.e. each combination of parameters, we:
1. Train the model with the current configuration;
2. Compute the configuration’s score with cross-validation;
3. Update best score and best parameters.

Why k-fold cross-validation?
If we simply split the training set into a sub-training set and a validation set, we would not use all
samples for the training, which could improve the model accuracy. To better exploit the available
data, we resort to the use of cross-validation.
We divide the training set in k equally long partitions and use sequentially k-1 of them as training
set and the remaining one as validation set. The cross-validation score is the average score of the k
validation scores.

Results
The scores obtained for all the configurations vary between 0.27 and 0.9.
Testing different value panels for hyperparameters, we observe that there exist several configurations
providing a good score, and that those configurations can be very different. For instance, the configuration ‘rbf, regularization = 4, min var = 0.001 provides good results, but so does the configuration
‘linear, regularization = 6, min var = 0.001.

Feature selection
This optimization of the parameters also acts
like a feature selection, since all the features
whose variances are above min var will be discarded. As shown in figure 7, we can see that
the algorithm has discarded 8 low-variance features.


Testing

Building the final classifier
Now that the best parameters are found, we can use them to build the final classifier.
1. The classifier is set with the best parameters previously computed;
2. The classifier is trained with the training set of 300 samples, previously used for cross-validation;
3. The classifier is tested on the testing set.

Test results
Running the algorithm, a test score = 0.89 is obtained, which is a satisfying result. We can run the
entire algorithm several times and check that the values are always close to 0.90.

Interpretation
The confusion matrix tells us what are the pairs of class that the algorithm solves
the best.
 Indeed, the sounds of classical and jazz music are sometimes similar since they both use classical
instruments. This explains those 6 misclassified jazz/classical samples.
 On the other side, both country and disco have drums and guitars and are easily distinguished
from classical, which explains the 0 misclassified disco/country samples for the classical class.
 The same happens for country and disco, that share a greater musical affinity between each
other than with classical or jazz. This explain those 4 misclassified disco/country samples.


Code

Code organization
In order to implement our music genre classification software, different design choices were taken in
account. We aimed at developing a modular system, that allowed us to test and validate various
components independently. Our product is designed to be as flexible and robust as possible, to grant
a wide degree of possibilities in terms of changing the configuration of the system.

Code highlights

Test Modules
In order to realize the extraction of the features, the automatic configuration of the
hyperparameters and the evaluation of the system, two different scripts are proposed:
test feature extraction.py and test classification.py.
Independent Extraction The extraction phase is performed inside test feature extraction.py;
this part of the program generates a csv file containing all the features from our data set. The extraction process could be really slow, compared to the evaluation phase, but, due to the modular
organization, it can be executed just once after all the features are defined.

Automatic Hyperparameters Estimation
After the feature extraction, various configurations
are tested inside test classification.py, with a k-fold approach, varying the hyperparameters.
The best configuration is saved as a json file and it will be used for the evaluation phase. This section
of the code, can be deactivated by a flag.

Flexible Features Definition
The module feature.py contains all the feature functions used
for our classification algorithm. All the functions are exposed to the rest of the program through
a python dictionary. Therefore adding a new function and the relative entry of the dictionary, is
enough to update the response of whole system, allowing for an easier testability and separation.
Debug Console Logs Our software informs the user about different aspects concerning the execution state, in order to give a clear picture of the progresses of the algorithm, highlighting the
execution time of every section

Conclusions
To conclude, we obtained an efficient genre classifier for the four selected genres, with a test error
equal to about 10%. The main errors occur when the classifier has to distinguish jazz from classical
and country from disco, which was predictable. The most useful features were the MFCC coefficients,
but all features must be selected to minimize the validation error.
This work not only results into a genre classifier for jazz, disco, country and classical music, but also
into a genre classifier constructor. The code could be used to build any n-genres classifier, since it
automatically optimizes its configuration according to the dataset.
Finally, the performances of the classifier could be improved with a larger dataset for the training,
more relevant features and more computational power for the parameters’ optimization. The number
of configuration to test is indeed growing exponentially with the value panels’ size of the parameters,
and quickly becomes a limiting factor.

