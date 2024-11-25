%%%% This script is published in conjunction with Dunkley Jones et al., 2020.
%%%% OPTiMAL: A new machine learning approach for GDGT-based
%%%% palaeothermometry. Climate of the Past. doi:10.5194/cp-16-2599-2020
%%%% Code and README housed at: https://github.com/carbonatefan/OPTiMAL

%%%% This file:
    %%%% Reads in a csv's (calibration dataset and ancient fossil dataset)
    %%%% Extracts the columns containing GDGT column headers
    %%%% Returns a csv file with Nearest Neighbour distances and
    %%%% temperature predictions from the GPR.
    %%%% Returns the learnt Sigma (lengthscale) values 
    %%%% Returns an array of distance estimates between each fossil and
    %%%% calibration datapoint (used to generate the calibration map)
    %%%% Returns a plot of the predicted error (1 standard deviation) vs.
    %%%% the nearest neighbour distances for the ancient dataset.
    %%%% Returns a plot of the predicted temperature with error bars (1
    %%%% standard deviation) vs. sample number.
    %%%% Returns a plot of the calibration data and how "far away" from the
    %%%% fossil data it lies. 
    %%%% Returns an excel spreadsheet containing all the outputs of this
    %%%% code.

%%%% Headers are specified for the calibraiton and fossil data to extract
%%%% the relevant values for the analysis. 

clear all
close all

%%%% ~~~~~~~~~~~~~~~~~~~~~~~ User inputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %%%

% Specify the calibration dataset
ancient_table=readtable('Example_Holocene.csv','ReadVariableNames',true);
% Specify the ancient dataset
calibration_table=readtable('Global_Calibration.csv','ReadVariableNames',true);

% Specify the column headers containing the geochemical data.
input_variables = ["GDGT_0",...
                   "GDGT_1",...
                   "GDGT_2",...
                   "GDGT_3",...
                   "Crenarchaeol",...
                   "Cren"];

% Specify the column header containing the environmental variable               
calibration_variable = ["Temp"]; 

%%%% ~~~~~~~~~~~~~~~~~~~~~~~ User inputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %%%

%%%% ~~~~~~~~~~~~~~~~~~~~~~~~~~ OPTiMAL  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %%%
% Please see function specific comments for details of the below code

% Initializes the structure with the calibration and ancient data 
[data_struct] = Initialize_Struct(calibration_table,...
                                  ancient_table);

% Processes the data in the structure

% Function inputs.....
% 1 = Plot figures of GP performance, 0 = no figures
see_local_function_figures = 1;
% D_nearest cut-off value, 0.5 after original work of OPTiMAL (2020)
D_nearest_cutoff = 0.5;

[data_struct] = Process_Struct_Gaussian(data_struct,...
                                        input_variables,...
                                        calibration_variable,...
                                        D_nearest_cutoff,...
                                        see_local_function_figures);

% Outputs the contents of the structure into an excel spreadsheet
Generate_Output(data_struct);

% List of all variables created and written out to the spreadsheet:
%%% Calibration_Dataset_matlab Sheet:
    %%% Once the Gaussian training stage is complete OPTiMAL
    %%% makes temperature predictions on the calibration to 
    %%% highlight model performance on known targets
        %%% Predicted_Temp: OPTiMAL SST prediction
        %%% Temp_low_95: SST prediction lower 95th percentile
        %%% Temp_upp_95: SST prediction upper 95th percentile
        %%% Temp_Stddev: SST prediction standard deviation
%%% Ancient_Dataset_matlab Sheet:
    %%% Ancient data with unknow SST's on which OPTiMAL makes predictions.
        %%% Predicted_Temp: OPTiMAL SST prediction
        %%% Temp_low_95: SST prediction lower 95th percentile
        %%% Temp_upp_95: SST prediction upper 95th percentile
        %%% Temp_Stddev: SST prediction standard deviation
        %%% D_Nearest: Closest distance between the lengthscale weighted
        %%%             ancient and calibration datapoints. Used as a QC
        %%%             measure where SST predictions are only considered 
        %%%             robust when D_Nearest <= 0.5
%%% Sigma_matlab Sheet:
    %%% List of the lengthscale learnt during the Gaussian Process. The
    %%% smaller the value the more temperature information is archived
    %%% within that dimension. Values are shown in order of the
    %%% input_variables outlined in the code
%%% Distance_Array_matlab Sheet:
    %%% A matrix of all D_Values (distance between the lengthscale weighted
    %%% ancient and calibration datapoints). The dimensions of the
    %%% matrix are the size of the calibration data x ancient data
    %%% (calibration = excel columns, ancient = excel rows). D_Nearest
    %%% represents the smallest value of a given row of data (i.e. the
    %%% closest calibration datapoint to a given ancient datapoint)

%%%% ~~~~~~~~~~~~~~~~~~~~~~~~~~ OPTiMAL  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %%%

%%%% ~~~~~~~~~~~~~~~~~~~~~~ Calibraiton Map  ~~~~~~~~~~~~~~~~~~~~~~~~ %%%

% Plotter function is commented out as it is a little slow. Data generated
% in the above code. Uncomment to see the calibration map.

% Plots a calibration map coloured by the D_values selected by the variable
% quant_choice

% Function inputs.....
% Quantile used to plot the global maps (float 0 to 1)
quant_choice = 0.5;
% Colourmap divisions: higher = more divisions of colour
divisions = 150;
% -999 flags the plotter to use the quant choice variable. A number returns
% a specific row of the data (i.e. a single ancient GDGT distribution)
data_options = -999;

% ~~~~~~ Uncomment to use ~~~~~~~~~~
% plotter(data_struct,...
%         quant_choice,...
%         divisions,...
%         data_options);
% ~~~~~~ Uncomment to use ~~~~~~~~~~

%%%% ~~~~~~~~~~~~~~~~~~~~~~ Calibraiton Map  ~~~~~~~~~~~~~~~~~~~~~~~~ %%%


% Below are the generalised functions that handle the GP. They are
% commented sufficiently (hopefully...) to enable an individual to tweak
% them to their needs and understand the implementation fo the GP. 

% ~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %

function [calibration_table] = Filter(calibration_table,...
                                      calibration_variable,...
                                      filter_value)
    
    % Choose the filter value and remove values in the calibration table that have that value
    remover =  table2array(calibration_table(:,calibration_variable)) == filter_value;
    calibration_table(remover,:) = [];

end

function [output_calibration, output_ancient, sigmaL, distance_array] =   Gaussian(calibration_table,...
                                                                                          ancient_table,...
                                                                                          input_variables,...
                                                                                          calibration_variable,...
                                                                                          D_nearest_cutoff,...
                                                                                          see_local_function_figures)
    
    % This function houses the code published in the original OPTiMAL paper
    % but packaged into a function
    
    % Outline of the function: 
    %   - Ingest the calibration data (with known environmental variable) and
    %   the ancient data (with environmental variable to be predicted).
    %   - Extract the standard deviation ( of the input variables.
    %   - Fit a gaussian processes regression to the calibration dataset
    %   (fitrgp). Parameters are set using the std of the inputs.
    %   - Check the GP performance on the calibration data.
    %   - Extract the GP calculated lengthscales.
    %   - Apply the GP to the ancient data.
    %   - Create the D_value array.
    %   - Format the outputs of the function
    %   - Check to see if you want to visualise the output with figures

    fprintf('\n~~~~~~~~~ Processing the Gaussian ~~~~~~~~\n')
                                                                                
    % Extract inputs using the input variables to slice the table
    ancient = ancient_table(:,input_variables);
    calibration = calibration_table(:,input_variables);
    calibration_var = calibration_table(:,calibration_variable);

    % Convert tables to arrays 
    ancient = table2array(ancient);
    calibration = table2array(calibration);
    calibration_var = table2array(calibration_var);
    
    % Prep the variables that feed into the GP
    gp_intput1 = std([calibration,calibration_var]);
    gp_intput2 = std(calibration_var);

    % Calibrate GP regression on full modern data set
    gprMdl = fitrgp(calibration,calibration_var,'KernelFunction','ardsquaredexponential',...
            'KernelParameters',gp_intput1,'Sigma',gp_intput2);  

    % Predict on the calibration data as a means of quantifying model
    % performance.
    [tempcalibration,tempcalibrationstd,tempcalibration95]=predict(gprMdl,calibration);

    % Calculate and save the learned lengthscales
    sigmaL = gprMdl.KernelInformation.KernelParameters(1:end-1);

    %Apply GP regression to ancient data set
    [tempancient,tempancientstd,tempancient95]=predict(gprMdl,ancient,'Alpha',0.05);
    
    % Calculate the D_value array. This is the Euclidean distance between
    % all points, but with each dimension scaled by the length scales
    
    %Get size of input variables list
    var_size = size(input_variables);
    var_stop = var_size(2);
    
    % Get the length of the input_variables list of strings
    var_stop = size(input_variables,2);
    % Create the empty array for distsq = (root(distance between points)^2)
    distsq_array = [];
    % Loop through all points in calibration and ancient and compare the distances between the two points
    for(i=1:length(ancient))
        for(j=1:length(calibration))
                dist=(calibration(j,1:var_stop)-ancient(i,1:var_stop))./sigmaL';
                distsq(j)=sqrt(sum(dist.^2));
        end
        % Full array
        distsq_array = [distsq_array;distsq];
        % Data for the dist_stats output
        [distmin(i),index(i)]=min(distsq);
        [distmax(i),indexmax(i)]=max(distsq);
        [distmean(i)]=mean(distsq);
        [distQ1(i)]=quantile(distsq,0.25);
        [distQ2(i)]=quantile(distsq,0.5);
        [distQ3(i)]=quantile(distsq,0.75);
    end

    % Create the distsq_array table
    distance_array = array2table(distsq_array);
    
    % Prep outputs
    Header_Ancient = ["Predicted_Temp","Temp_low 95","Temp_upp_95","Temp_Stddev","D_Nearest"];
    Header_Calibration = ["Predicted_Temp","Temp_low_95","Temp_upp_95","Temp_Stddev"];

    % Copy in the model temperature outputs
    temp_ancient = table(tempancient, tempancient95(:,1), tempancient95(:,2), tempancientstd, distmin.');
    temp_calibration = table(tempcalibration, tempcalibration95(:,1), tempcalibration95(:,2), tempcalibrationstd);

    % Create the table headers
    temp_ancient.Properties.VariableNames = Header_Ancient;
    temp_calibration.Properties.VariableNames = Header_Calibration;

    % Join Tables into final data format
    output_ancient = [ancient_table,temp_ancient];
    output_calibration = [calibration_table,temp_calibration];
    
    % Checks to see if you want these data plotting
    if see_local_function_figures == 1
        %%%%%% Visualise the data %%%%%%%%
        %Create figure - OPTiMAL SST standard error vs. DNearest
        figure
        set(gca, 'FontSize', 12); 
        semilogx(distmin.',tempancientstd,'.', 'MarkerSize', 16); hold on;
        plot(D_nearest_cutoff*ones(size([3:9])),[3:9],'k:', 'LineWidth', 2); hold off;
        grid on
        xlabel('$D_\mathrm{nearest}$','Interpreter', 'latex')
        ylabel(['St. Dev. OPTiMAL SST (' char(176) 'C)'])
        saveas(gcf,"plot1.png")

        %Create figure - OPTiMAL SST vs. sample number
        figure
        SampleNumber=1:length(tempancient);
        set(gca, 'FontSize', 12);
        hold on
        for (j=1:length(SampleNumber)),
            if (distmin(j)<D_nearest_cutoff),
                plot([SampleNumber(j),SampleNumber(j)], [tempancient(j)-tempancientstd(j),tempancient(j)+tempancientstd(j)],'-k','LineWidth',1),
            else
                plot([SampleNumber(j),SampleNumber(j)], [tempancient(j)-tempancientstd(j),tempancient(j)+tempancientstd(j)],'-','color', [0.8 0.8 0.8], 'LineWidth',0.5),
            end
        end
        scatter(SampleNumber(distmin>=D_nearest_cutoff),tempancient(distmin>=D_nearest_cutoff), 15, [0.8 0.8 0.8], 'filled');
        scatter(SampleNumber(distmin<D_nearest_cutoff),tempancient(distmin<D_nearest_cutoff), 25, (distmin(distmin<D_nearest_cutoff)), 'filled');
        c = colorbar;
        xlabel('Sample Number'),
        ylabel(['OPTiMAL SST (' char(176) 'C)']);
        c.Label.String = 'D_{nearest}';
        saveas(gcf,"plot2.png")

    end
   
end

function [data_struct] = Initialize_Struct(calibration_table,...
                                           ancient_table)

    % Takes two tables of information and places them into a struct
                                 
    fprintf('\n~~~~~~~~~ Building Struct ~~~~~~~~\n')

    data_struct.("Calibration_Data") = calibration_table;
    data_struct.("Ancient_Data") = ancient_table;

end

function [data_struct] = Process_Struct_Gaussian(data_struct,...
                                                 input_variables,...
                                                 calibration_variable,...
                                                 D_nearest_cutoff,...
                                                 see_local_function_figures)

    % Takes a struct containing the calibration data and fossil data and
    % completes the Gaussian Process analysis on it. Returns the struct but with
    % filled in Gaussian predictions/stats, sigma lengthscales and also the
    % distance array
    
    fprintf('\n~~~~~~~~~ Process Structures ~~~~~~~~\n')
    
    % Setting the input again so they can be passed onwards to the Gaussian
    % Process below
    
    input_variables = input_variables;
    calibration_variable = calibration_variable;
    D_nearest_cutoff = D_nearest_cutoff;

    calibration_field = data_struct.Calibration_Data;
    ancient_field = data_struct.Ancient_Data;
    
    % Runs the Gaussian process 
    [output_calibration, output_ancient, sigmaL, distance_array] = Gaussian(calibration_field,...
                                                                            ancient_field,...
                                                                            input_variables,...
                                                                            calibration_variable,...
                                                                            D_nearest_cutoff,...
                                                                            see_local_function_figures);
    
    % Writes out the various bits of interest into the structure
    data_struct.Calibration_Data = output_calibration;
    data_struct.Ancient_Data = output_ancient;        
    data_struct.("SigmaL") = sigmaL;
    data_struct.("Distance_Array") = distance_array;    

end


function plotter(data_struct,...
                 quant_choice,...
                 divisions,...
                 data_options)

    % Takes the processed structure and plots the data in a global view
    % whereby the datapoints plotted are the calibration data coloured by
    % the gaussian estimate of distance to the fossil data. Can specify the
    % quant_choice variables to choose the quantile you would like to
    % visualise (e.g. 0.25 = Q2, 0.5 = Median, 0.75 = Q3 etc.)
    % In this example, Median D_Values are used, quant_choice = 0.5
    
    fprintf('\n~~~~~~~~~ Plotting Data ~~~~~~~~\n')
    
    % Start figure making
    figure
    % Extract the spatial data and the distance array
    table = data_struct.Calibration_Data;
    lat = table2array(table(:,"Latitude"));
    lon = table2array(table(:,"Longitude"));
    dist_array = data_struct.Distance_Array;
    dist_array = table2array(dist_array);
    
    if data_options == -999
            dist_array = data_struct.Distance_Array;
            dist_array = table2array(dist_array);
            % Extract the quantile from the distance array
            for i = 1:(size(dist_array,2))
                col = dist_array(:,i);
                output(i) = mean(col);
                quant(i) = quantile(col,quant_choice);

            end
    else
            dist_array = data_struct.Distance_Array;
            dist_array = table2array(dist_array);
            quant = dist_array(data_options,:);
    end
    
    % Plot the data
    geoscatter(lat,lon,50,quant, 'filled', 'MarkerEdgeColor','black');
    
    % Colourmap options
    % Diverging colormap options. Here it is a Green - Yellow - Red
    % colourmap. Can specify divisions variable to adjsut the colour
    % binning
    up = round(divisions/2);
    down = divisions - up;
    
    % Green end values: 75, 184, 157
    % Mid Values: 240, 245, 186
    % Red end values: 173, 75, 62
    
    r_up = linspace(75,240,up);
    g_up = linspace(184,245,up);
    b_up = linspace(157,186,up);
    
    r_down = linspace(240,173,down);
    g_down = linspace(245,75,down);
    b_down = linspace(186,62,down);
    
    r_down = r_down(2:end);
    g_down = g_down(2:end);
    b_down = b_down(2:end);
    
    r = [r_up, r_down];
    g = [g_up, g_down];
    b = [b_up, b_down];
    
    rgb = [r.',g.',b.']/255;
    
    % Apply Colourmap
    colormap(rgb);
    c = colorbar;
    c.Label.String = sprintf("Nearness");
    % Put topographic basemap on
    geobasemap topographic 
    saveas(gcf,"plot3.png")

end


function Generate_Output(data_struct)
    
    % Writes out the struct where each field is a excel sheet
    
    fprintf('\n~~~~~~~~~ Saving Output ~~~~~~~~\n')

    excelFilename = 'OPTiMAL_Output.xlsx';
    writetable(data_struct.Calibration_Data, excelFilename, 'Sheet', sprintf('%s_matlab', "Calibration_Data"));
    writetable(data_struct.Ancient_Data, excelFilename, 'Sheet', sprintf('%s_matlab', "Ancient_Data"));
    writematrix(data_struct.SigmaL, excelFilename, 'Sheet', sprintf('%s_matlab', "SimgaL"));
    writetable(data_struct.Distance_Array, excelFilename, 'Sheet', sprintf('%s_matlab', "Distance_Array"));

end

