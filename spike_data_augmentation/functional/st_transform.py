st_transform(e, S, T, Roll, imDim):
    ## inputs:
    ## e = input event matrix, shape: N,4
    ## S = spatial transform matrix, shape: 3,3
    ## T = time transform vector, shape: 2, ([0]: scale amount, [1]: shift amount)
    ## Roll = bool to determine if out of range transformed events are rolled into plane on opposite
    ## imDim = HxW, shape = 2,
    ## output:
    ## st_e = spatial and temporal transformed event matrix, shape: Nx4
    
    nevt = e.shape[0]
    eAddr = e[:,0:2]
    ones_vec = np.ones((nevt,1))
    eAddrHomog = np.hstack((eAddr[:,0].reshape(nevt,1),eAddr[:,1].reshape(nevt,1),ones_vec))
    #spatial transform
    s_Res = np.matmul(S,eAddrHomog.T)
    #Roll coordinates if specified 
    s_e_X = s_Res[0,:]
    s_e_Y = s_Res[1,:]
    s_Res_oorX_P = np.where(s_Res[0,:] >= imDim[1]) # Out Of Range coordinates based on imDim
    s_Res_oorX_N = np.where(s_Res[0,:] < 0)

    s_Res_oorY_P = np.where(s_Res[1,:] >= imDim[0])
    s_Res_oorY_N = np.where(s_Res[1,:] < 0)

    if Roll:
        s_e_X[s_Res_oorX_P] = s_e_X[s_Res_oorX_P]-imDim[0] # Roll X right 
        s_e_X[s_Res_oorX_N] = s_e_X[s_Res_oorX_N]+imDim[0] # Roll X left
        s_e_Y[s_Res_oorY_P] = s_e_Y[s_Res_oorY_P]-imDim[1] # Roll Y down
        s_e_Y[s_Res_oorY_N] = s_e_Y[s_Res_oorY_N]+imDim[1] # Roll Y up
    else:
        s_e_X[s_Res_oorX_P] = imDim[0] # Clip X pos.
        s_e_X[s_Res_oorX_N] = 0 # Clip X neg.
        s_e_Y[s_Res_oorY_P] = imDim[1] # Clip Y pos.
        s_e_Y[s_Res_oorY_N] = 0 # Clip Y neg.
        
    s_e = np.vstack((s_e_X,s_e_Y))
    
    #scale time 
    t_v = e[:,3]
    dT = np.diff(t_v)
    s_dT = T[0]*dT
    sh_dT = s_dT + T[1]
    
    #if the times are shifted beyond 0, roll time value 
    badT = np.where(sh_dT < 0)
    sh_dT[badT] = np.max(t_v)+ sh_dT[badT]
    t_v_new = t_v[1:]+sh_dT
    t_v_new = np.append(t_v[0], t_v_new)
    p = e[:,2]
    
    eTransf = np.vstack((s_e_X, s_e_Y, p.T, t_v_new))
    
    return eTransf.T
