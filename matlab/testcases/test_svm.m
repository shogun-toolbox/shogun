function y = test_svm(filename)

  eval(filename);

  kname = ['set_kernel ', 'GAUSSIAN REAL ', num2str(size_), ' ',  num2str(width_)];
  sg('set_features', 'TRAIN', traindat);
  sg('set_labels', 'TRAIN', labels);
  sg('send_command', kname);
  sg('send_command', 'init_kernel TRAIN');
%  sg('send_command', 'new_svm LIBSVM');
  sg('send_command', 'new_svm LIGHT');
  sg('send_command','svm_epsilon 1e-6')
  sg('send_command', 'c 10');
  tic;
  sg('send_command', 'svm_train');
%  time_light(1)=toc
  [bias, testalphas]=sg('get_svm');
  %o3=sg('get_svm_objective');

  t = testalphas(:,1)'
%  size(t)
%  size(alphas)
t
alphas
  a = max(max(abs(alphas-t)))
  if(a<1e-7)
    y = 0;
  else
    y = 1;
  end


