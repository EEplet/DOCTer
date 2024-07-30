# DOCTer
The code of the paper "DOCTer: A Novel EEG-based Diagnosis Framework for Disorders of Consciousness".  
DOCTer is a framework designed for diagnosing disorders of consciousness(DOC) using EEG through deep learning methods. For more information about the DOCTer framework, please refer to our paper.

## Dependencies
- Python 3.6+
- mne 1.1.0+
- pycrostates 0.3.0+
- PyTorch 0.4.0+


## Example
Here is a simple example of how to use the DOCTer framework:

```bash
epoch=100
datapath="/data/EEG/"
seed=99
fold=10
chs='all'
testf=20
csvfile='./res.csv'

python master_old.py --normalize "y" --chs 'all' --testfreq $testf --csvfile $csvfile --fold $fold --timelen -1 --datapath $datapath --seed 99 --dropout 0.4 --weight_decay 0.0001 --epochs $epoch --batch_size 256 --lr 0.001 --clip 100 --model "DOCTer"  --cuda "cuda:0"
```

## Notes
Thank you for visiting. Due to time constraints, there are some imperfections. We will continue to improve this repository in the future.

## Contact
For questions, feedback, or suggestions, please contact us at:

- Email: yue.cao@zju.edu.cn
- GitHub Issues: [https://github.com/EEplet/DOCTer/issues](https://github.com/EEplet/DOCTer/issues)

## Citation
If you find our code is useful, please cite our paper.

