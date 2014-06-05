from __future__ import division, print_function, absolute_import 

from pnet.vzlog import default as vz
import numpy as np
import amitgroup as ag
import itertools as itr
import sys
import os
import gv


import pnet
import time

def test(ims, labels, net):
    yhat = net.classify(ims)
    return yhat == labels

if pnet.parallel.main(__name__): 
    ag.set_verbose(True)
    print("1")
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('seed', metavar='<seed>', type=int, help='Random seed')

    args0 = parser.parse_args()

    #for i in xrange(1, 7):
    #    print(make_support(i, 4).astype(np.uint8))

    #params = randomize_layers_parameters(args0.seed)
    #print(params)

    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []

    print("2")
    # Switch which experiment here
    #from pnet.mnist_danny import parse_background_random as loadf
    #from pnet.mnist_danny import parse_background_images as loadf
    #from pnet.mnist_danny import parse_basic as loadf
    print("Loading...")
    #mnist_data = loadf()

    #mnist_data = np.load(os.path.expandvars('$RAND_MNIST/data-random.npy')).flat[0]
    #mnist_data = np.load(os.path.expandvars('$BKG_MNIST/data-image-training.npy')).flat[0]
    #mnist_data = np.load(os.path.expandvars('$ROT_MNIST/data-rotation-training.npy')).flat[0]
    mnist_data = np.load(os.path.expandvars('$ROTBKG_MNIST/data-rot-bkg.npy')).flat[0]
    print("Done.")

    if 0: 
        from pnet.vzlog import default as vz
        import gv
        for i in xrange(10):
            gv.img.save_image(vz.generate_filename(), mnist_data['training_image'][i])

        import sys; sys.exit(0)

    for training_seed in xrange(1):
        print("Inside")
        net = pnet.PartsNet.load('basic-rot-net3.npy')

        #TRAIN_SAMPLES = 12000 
        #TRAIN_SAMPLES = 1200 

        digits = range(10)
        #ims = ag.io.load_mnist('training', selection=slice(0 + 3000 * training_seed, TRAIN_SAMPLES + 3000 * training_seed), return_labels=False)
        #ims = mnist_data['training_image'][0 + 1000 * training_seed : TRAIN_SAMPLES + 1000 * training_seed]
        print('Extracting subsets...')
        ims10k = mnist_data['training_image'][:10000]
        label10k = mnist_data['training_label'][:10000]
        #ims10k = mnist_data['training_image'][:100]
        #label10k = mnist_data['training_label'][:100]

        #ims10k = ims10k[label10k <= 1]
        #label10k = label10k[label10k <= 1]

        ims2k = mnist_data['training_image'][10000:12000]
        label2k = mnist_data['training_label'][10000:12000]
        print('Done.')

        #print(net.sizes(X[[0]]))

        start0 = time.time()
        print('Training unsupervised...')
        net.train(ims10k)
        print('Done.')
        end0 = time.time()

        #print("Extracting edges")
        #edges = net.layers[0].extract(ims10k)
        
        net.save('tmp{}-rot2-basic.npy'.format(training_seed))

        sup_ims = []
        sup_labels = []
        # Load supervised training data
        for d in digits:
            ims0 = ims10k[label10k == d]
            sup_ims.append(ims0)
            sup_labels.append(d * np.ones(len(ims0), dtype=np.int64))

        sup_ims = np.concatenate(sup_ims, axis=0)
        sup_labels = np.concatenate(sup_labels, axis=0)


        for classifier in ['rot-mixture']:
            for rotspread in [1, 0]:
                #net.layers[0]._settings['rotation_spreading_radius'] = rotspread

                print('Classifier:', classifier, 'Rotational spreading:', rotspread)
                n_orientations = 16
                n_comp = 5
                if classifier == 'rot-mixture':
                    cl = pnet.RotationMixtureClassificationLayer(n_components=n_comp, n_orientations=n_orientations, min_prob=1e-5, pooling_settings=dict(
                            shape=(4, 4),
                            strides=(4, 4),
                            rotation_spreading_radius=rotspread,
                        ))
                else:
                    pass

                clnet = pnet.PartsNet([net, cl])
                        
                start1 = time.time()
                print('Training supervised...')
                clnet.train(sup_ims, sup_labels)
                print('Done.')
                end1 = time.time()

            
                # Plot model
                all_comps = clnet.layers[-1]._extra['training_comp'] 

                grid = pnet.plot.ImageGrid(len(digits), n_comp, ims10k.shape[1:])


                try:
                    from skimage.transform import rotate
                    # Rotate the ims10k files
                    print("Rotating all of them")
                    from pylab import cm



                    for d in digits:
                        comps = all_comps[d]

                        ims10k_d = ims10k[label10k == d]

                        rot_ims10k = np.asarray([[rotate(im, -rot, resize=False) for rot in np.arange(n_orientations) * 360 / n_orientations] for im in ims10k_d])

                        visparts = np.asarray([
                            rot_ims10k[comps[:,0]==k,comps[comps[:,0]==k][:,1]].mean(0) for k in xrange(n_comp)
                        ])

                        XX = [
                            rot_ims10k[comps[:,0]==k,comps[comps[:,0]==k][:,1]] for k in xrange(n_comp)
                        ]

                        M = 50

                        grid0 = pnet.plot.ImageGrid(n_comp, min(M, np.max(map(len, XX))), ims10k.shape[1:])
                        for k in xrange(n_comp):
                            for i in xrange(min(M, len(XX[k]))):
                                grid0.set_image(XX[k][i], k, i, vmin=0, vmax=1, cmap=cm.gray)
                        grid0.save(vz.generate_filename(), scale=3)

                        for k in xrange(n_comp):
                            grid.set_image(visparts[k], d, k, vmin=0, vmax=1, cmap=cm.gray)

                    grid.save(vz.generate_filename(), scale=5)
                except:
                    print("RAISE")
                    pass


                #print("Now testing...")
                ### Test ######################################################################

                corrects = 0
                total = 0
                if 0:
                    test_ims, test_labels = mnist_data['test_image'], mnist_data['test_label']
                else:
                    test_ims = ims2k
                    test_labels = label2k


                with gv.Timer("Split to batches"):
                    ims_batches = np.array_split(test_ims, 50)
                    labels_batches = np.array_split(test_labels, 50)

                def format_error_rate(pr):
                    return "{:.2f}%".format(100*(1-pr))

                import gv
                #with gv.Timer('Testing'):
                start2 = time.time()
                args = (tup+(clnet,) for tup in itr.izip(ims_batches, labels_batches))
                print("Testing now...")
                for i, res in enumerate(pnet.parallel.starmap(test, args)):
                    corrects += res.sum()
                    total += res.size
                    pr = corrects / total
                print("Testing done.")
                end2 = time.time()

                error_rate = 1.0 - pr 


                num_parts = 0#net.layers[1].num_parts

                error_rates.append(error_rate)
                print(training_seed, 'error rate', error_rate * 100, 'num parts', num_parts)#, 'num parts 2', net.layers[3].num_parts)

                unsup_training_times.append(end0 - start0)
                sup_training_times.append(end1 - start1)
                testing_times.append(end2 - start2)

                #print('times', end0-start0, end1-start1, end2-start2)

                all_num_parts.append(num_parts)

                #vz.section('MNIST')
                #gv.img.save_image(vz.generate_filename(), test_ims[0])
                #gv.img.save_image(vz.generate_filename(), test_ims[1])
                #gv.img.save_image(vz.generate_filename(), test_ims[2])

                # Vz
                #net.infoplot(vz)


    if 0:
        print(r"{ppl} & {depth} & {num_parts} & {unsup_time:.1f} & {test_time:.1f} & ${rate:.2f} \pm {std:.2f}$ \\".format(
            ppl=2,
            depth=maxdepth,
            num_parts=r'${:.0f} \pm {:.0f}$'.format(np.mean(all_num_parts), np.std(all_num_parts)),
            unsup_time=np.median(unsup_training_times) / 60,
            #sup_time=np.median(sup_training_times),
            test_time=np.median(testing_times) / 60,
            rate=100*np.mean(error_rates),
            std=100*np.std(error_rates)))

        print(r"{ppl} {depth} {num_parts} {unsup_time} {test_time} {rate} {std}".format(
            ppl=2,
            depth=maxdepth,
            num_parts=r'${:.0f} \pm {:.0f}$'.format(np.mean(all_num_parts), np.std(all_num_parts)),
            unsup_time=np.median(unsup_training_times) / 60,
            #sup_time=np.median(sup_training_times),
            test_time=np.median(testing_times) / 60,
            rate=100*np.mean(error_rates),
            std=100*np.std(error_rates)))


        #np.savez('gdata2-{}-{}-{}.npz'.format(maxdepth, split_criterion, split_entropy), all_num_parts=all_num_parts, unsup_time=unsup_training_times, test_time=testing_times, rates=error_rates)

        print('mean error rate', np.mean(error_rates) * 100)
        #net.save(args.model)
