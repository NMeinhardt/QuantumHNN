% Returns index and the probability of the most likely outcome of a
% Z-measurement, given a probability vector prob. Prob must contain
% probabilities of all comp basis states and starts with |0>.
%  If several oucomes are equally likely, all corresponding indices are
%  returned as a vector.
function [indices, max_prob] = MostLikelyOutcome(prob)

max_prob = max(prob);
indices = find(prob==max_prob);

end 