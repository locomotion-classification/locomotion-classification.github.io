function [X, Y] = trace_condition(traces,length_traces,key,start_index,stop_index,randomize)
    %[X, Y] = trace_condition(traces,length_traces,key,start_index,stop_index,randomize)
    %Conditions data for feeding into MATLAB's NN framework
    traces = traces(:,1:length_traces);
    [r, c] = size(traces);
    traces = traces';
    key = categorical(double(key(:,1)));
    for i = start_index:stop_index
        X(:,:,:,i-start_index+1) = reshape(traces(:,i),[c 1 1]);
    end
    Y = key(start_index:stop_index,1);
    if randomize == 1
        idx = randperm(length(Y));
        X = X(:,:,:,idx);
        Y = Y(idx,1);
    else %do nothing
    end
    disp('trace conditioning completed');
end