from __future__ import absolute_import
from collections import OrderedDict
from .layers import *
from .roi import *

def profiling(net, input=None):
    # input is either a Blob with the shape of (batch,h,w,c) or a dict of them
    layers=[]
    if isinstance(input,dict):
        blob_dict = OrderedDict(input)
        not_ref = [input[k] for k in input]
    else:
        blob_dict = OrderedDict({'data': input})
        not_ref=[input]
    for i, layer in enumerate(net.net.layer):
        out = None
        if len(layer.top) == 1 and len(layer.bottom) == 1:
            if layer.type == 'Convolution':
                param = layer.convolution_param
                ksize = param.kernel_size
                if len(ksize) == 0:
                    kh = param.kernel_h
                    kw = param.kernel_w
                else:
                    kh = param.kernel_size[0]
                    kw = kh
                if len(param.pad)==0: 
                    ph = param.pad_h
                    pw = param.pad_w
                else:
                    ph = param.pad[0]
                    pw = ph
                if len(param.stride)==0 and not(param.stride_h ==0 and param.stride_w==0):
                    sh = param.stride_h
                    sw = param.stride_w
                elif len(param.stride)!=0:
                    sh = param.stride[0]
                    sw = sh
                else:
                    sh = 1
                    sw = 1
                print("_________________________________________________________________________")
                print("kernel size type: ", type(param.kernel_size))
                print("_________________________________________________________________________")
                print("pad size type: ", type(param.pad))
                print("_________________________________________________________________________")
                print("stride size type: ", type(param.stride))
                print("_________________________________________________________________________")
                #try:
                   # out = Conv(blob_dict[layer.bottom[0]], [param.kernel_size[0], param.kernel_size[0]], param.num_output, param.stride,
                out = Conv(blob_dict[layer.bottom[0]], [kh, kw], param.num_output, [sh, sw],
                             [ph, pw], None, layer.name, group_size=param.group)
                #except:
                #    print(ksize, type(ksize))
            if layer.type == 'InnerProduct':
                param=layer.inner_product_param
                out= fc(blob_dict[layer.bottom[0]],param.num_output,None,layer.name)
            if layer.type == 'ReLU':
                out = Activation(blob_dict[layer.bottom[0]], 'relu', layer.name)
            if layer.type == 'PReLU':
                out = Activation(blob_dict[layer.bottom[0]], 'prelu', layer.name)
            if layer.type == 'Pooling':
                param = layer.pooling_param
                out = Pool(blob_dict[layer.bottom[0]], param.kernel_size, param.stride,
                             param.pad, layer.name,param.pool,ceil=True)
            if layer.type == 'Normalize':
                out = Norm(blob_dict[layer.bottom[0]], 'norm', layer.name)
            if layer.type == 'BatchNorm':
                out= Norm(blob_dict[layer.bottom[0]],'batch_norm',layer.name)
            if layer.type== 'LRN':
                out= Norm(blob_dict[layer.bottom[0]],'lrn',layer.name)
            if layer.type == 'Permute':
                shape=[blob_dict[layer.bottom[0]][dim-1] for dim in layer.permute_param.order[1:]]
                out = Permute(blob_dict[layer.bottom[0]],shape,layer.name)
            if layer.type == 'Flatten':
                out = Flatten(blob_dict[layer.bottom[0]], layer.name)
            if layer.type == 'Scale':
                out =Scale (blob_dict[layer.bottom[0]], name = layer.name)
            if layer.type == 'Softmax':
                out =Softmax (blob_dict[layer.bottom[0]], name = layer.name)
            if layer.type == 'Dropout':
                out =Dropout (blob_dict[layer.bottom[0]], name = layer.name)
            if layer.type == 'Reshape':
                out =Reshape (blob_dict[layer.bottom[0]],shape=layer.reshape_param.shape.dim, name = layer.name)
            if out:
                try:
                    not_ref.remove(blob_dict[layer.bottom[0]])
                except:
                    pass
                blob_dict[layer.top[0]] = out()
                not_ref.append(blob_dict[layer.top[0]])
                layers.append(out)
            else:
                assert 'layer type: %s cannot be P' % (layer.type)
        elif len(layer.bottom)>1:
            # for multi input layer
            if layer.type=='Eltwise':
                param=layer.eltwise_param
                out = Eltwise([blob_dict[bottom] for bottom in layer.bottom],
                              type=param.EltwiseOp.Name(param.operation),name=layer.name)
            if layer.type=='PSROIPooling':
                param=layer.psroi_pooling_param
                out = PSROIPool(blob_dict[layer.bottom[0]],blob_dict[layer.bottom[1]],
                                param.output_dim,param.group_size)
            if layer.type=='ROIPooling':
                param=layer.roi_pooling_param
                out = ROIPool(blob_dict[layer.bottom[0]],blob_dict[layer.bottom[1]],
                              param.pooled_w,param.pooled_h,layer.name)
            if layer.type == "Concat":
                param = layer.concat_param
                out = Concat([blob_dict[bottom] for bottom in layer.bottom],param.axis,layer.name)
            if out:
                for bottom in layer.bottom:
                    try:
                        not_ref.remove(blob_dict[bottom])
                    except:
                        pass
                blob_dict[layer.top[0]] = out()
                not_ref.append(blob_dict[layer.top[0]])
                layers.append(out)
            else:
                assert 'layer type: %s cannot be P' % (layer.type)
        elif len(layer.top)>1:
            if layer.type == 'Slice':
                param=layer.slice_param
                out =Slice (blob_dict[layer.bottom[0]], name = layer.name,slice_point=param.slice_point,axis=param.axis)
            if out:
                try:
                    not_ref.remove(blob_dict[layer.bottom[0]])
                except:
                    pass
                for o,top in zip(out(),layer.top):
                    blob_dict[top] = o
                    not_ref.append(blob_dict[top])
                layers.append(out)
    return blob_dict,layers
