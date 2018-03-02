import tensorflow as tf
import numpy as np

def MANN(n_inputs, n_hnodes, n_outputs, mem_size, mem_dim, inputs, batch_size=10, n_reads=4):

	#
	#	MODULE WEIGHTS
	#

	n_xh = n_inputs+n_hnodes
	n_rd = n_reads*mem_dim
	n_hr = n_hnodes+n_rd
	gamma = 0.95
	W_init = tf.contrib.layers.xavier_initializer()
	b_init = tf.constant_initializer(0)

	with tf.variable_scope('Weights'):
		# LSTM Gate Weights
		W_gf = tf.get_variable('W_forget_gate', shape=[n_xh,n_hnodes], initializer=W_init)
		b_gf = tf.get_variable('b_forget_gate', shape=[n_hnodes], initializer=b_init)
		W_gi = tf.get_variable('W_input_gate', shape=[n_xh,n_hnodes], initializer=W_init)
		b_gi = tf.get_variable('b_input_gate', shape=[n_hnodes], initializer=b_init)
		W_go = tf.get_variable('W_output_gate', shape=[n_xh,n_hnodes], initializer=W_init)
		b_go = tf.get_variable('b_output_gate', shape=[n_hnodes], initializer=b_init)
		# LSTM Tanh Weights
		W_u = tf.get_variable('W_remember', shape=[n_xh,n_hnodes], initializer=W_init)
		b_u = tf.get_variable('b_remember', shape=[n_hnodes], initializer=b_init)
		# Controller Weights
		W_kr = tf.get_variable('W_read_key', shape=[n_hnodes,n_rd], initializer=W_init)
		b_kr = tf.get_variable('b_read_key', shape=[n_rd], initializer=b_init)
		W_kw = tf.get_variable('W_write_key', shape=[n_hnodes,n_rd], initializer=W_init)
		b_kw = tf.get_variable('b_write_key', shape=[n_rd], initializer=b_init)
		W_ga = tf.get_variable('W_sigmoid_alpha', shape=[n_hnodes,n_reads], initializer=W_init)
		b_ga = tf.get_variable('b_sigmoid_alpha', shape=[n_reads], initializer=b_init)
		# Logit Weights
		W_o = tf.get_variable('W_output', shape=[n_hr,n_outputs], initializer=W_init)
		b_o = tf.get_variable('b_output', shape=[n_outputs], initializer=b_init)
		pass

	thetas = [W_gf,b_gf,W_gi,b_gi,W_go,b_go,W_u,b_u,W_kr,b_kr,W_kw,b_kw,W_ga,b_ga,W_o,b_o]

	#
	#	INITIAL MEMORY STATES
	#

	def tfloat32(x, name=''):
		return tf.Variable(tf.cast(np.asarray(x, dtype=np.float32), tf.float32), name=name, trainable=False)

	M_t0 = tfloat32(1e-6*np.random.rand(batch_size,mem_size,mem_dim),'memory_t0')
	h_t0 = tfloat32(np.zeros((batch_size,n_hnodes)),'hidden_state_t0')
	c_t0 = tfloat32(np.zeros((batch_size,n_hnodes)),'memory_state_t0')
	wu_t0 = tfloat32(np.zeros((batch_size,mem_size)),'usage_weight_t0')
	wr_t0 = tfloat32(np.zeros((batch_size,n_reads,mem_size)),'read_weight_t0')
	r_t0 = tfloat32(np.zeros((batch_size,n_reads,mem_dim)),'read_vector_t0')

	st8_t0 = (M_t0, h_t0, c_t0, wu_t0, wr_t0, r_t0)

	#
	#	FORWARD ITERATION
	#

	def forward(st8_tm1,X_t):

		M_tm1, h_tm1, c_tm1, wu_tm1, wr_tm1, r_tm1 = st8_tm1

		X_t_r = tf.reshape(X_t,[-1,n_inputs])
		xh = tf.concat([X_t_r,h_tm1],1)
		gf = tf.sigmoid(tf.matmul(xh,W_gf)+b_gf)
		gi = tf.sigmoid(tf.matmul(xh,W_gi)+b_gi)
		go = tf.sigmoid(tf.matmul(xh,W_go)+b_go)

		u_t = tf.tanh(tf.matmul(xh,W_u)+b_u)
		c_t = c_tm1*gf + u_t*gi
		h_t = c_t*go

		kr_t = tf.reshape(tf.tanh(tf.matmul(c_t,W_kr)+b_kr),[batch_size,n_reads,mem_dim])
		kw_t = tf.reshape(tf.tanh(tf.matmul(c_t,W_kw)+b_kw),[batch_size,n_reads,mem_dim])

		k_norm = tf.norm(kr_t,axis=2,keep_dims=True)
		m_norm = tf.norm(M_tm1,axis=2,keep_dims=True)
		inner_prod = tf.matmul(kr_t,tf.transpose(M_tm1,perm=[0,2,1]))
		norm_prod = tf.matmul(k_norm,tf.transpose(m_norm,perm=[0,2,1]))
		wr_t = tf.nn.softmax(inner_prod/norm_prod)
		wu_1 = wu_tm1*gamma + tf.reduce_sum(wr_t,1)
		r_t = tf.matmul(wr_t,M_tm1)

		ga = tf.expand_dims(tf.sigmoid(tf.matmul(h_t,W_ga)+b_ga),2)
		_, wlu_inds = tf.nn.top_k(-wu_1,k=n_reads)
		wlu_t = tf.reduce_sum(tf.one_hot(wlu_inds,depth=mem_size),1,keep_dims=True)
		ww_t = wr_t*ga + wlu_t*(1-ga)
		wu_t = wu_1 + tf.reduce_sum(ww_t,1)
		M_1 = tf.multiply(M_tm1,tf.transpose(-wlu_t,perm=[0,2,1])+1)
		M_t = M_1 + tf.matmul(tf.transpose(ww_t,perm=[0,2,1]),kw_t)

		st8_t = (M_t, h_t, c_t, wu_t, wr_t, r_t)
		return st8_t

	# input to this function should be shaped (batch_size,sequence_length,n_inputs) btw
	st8_tf = tf.scan(forward,elems=inputs,initializer=st8_t0,parallel_iterations=100)
	M_f, h_f, c_f, wu_f, wr_f, r_f = st8_tf

	hr = tf.concat([h_f,tf.reshape(r_f,[-1,batch_size,n_rd])],2)
	o_f = tf.tensordot(hr,W_o,1)+b_o
	return M_f, h_f, c_f, wu_f, wr_f, r_f, o_f