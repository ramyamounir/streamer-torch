import json, os, shutil, copy
import numpy as np
from glob import glob


class Annotation(object):
    def __init__(self,start,end,colour,action):
        self.start = start
        self.end = end
        self.colour = colour
        self.action = action

        self.children = []
        self.contenders = []
        self.contender_ious = []
        self.gt_match_iou = 0.0

    def iou(self, o):
        intersection = max(0, min(self.end, o.end) - max(self.start, o.start))
        union = max(self.end, o.end) - min(self.start, o.start)
        return intersection / union

    def to_dict(self):
        return {
          'start': self.start,
          'end': self.end,
          'colour': self.colour,
          'action': self.action
        }

class AnnotationsAsGraph(object):
    ground_truth = []
    graph = None

    @staticmethod
    def file_to_obj(annotations_file, merge_annots=True):

        def merge_gt(gt):
            new_gt = []
            for r_i, r in enumerate(gt):
                if r_i == 0:
                    new_gt.append(copy.deepcopy(r))
                    continue
                if new_gt[-1].action == r.action or new_gt[-1].action == f'still {r.action}':
                    new_gt[-1].end = r.end
                else:
                    new_gt.append(copy.deepcopy(r))
            return new_gt

        def get_final_layers(layers):
            num_layers = len(layers)
            idxs = [0] * num_layers

            def get_children(layer_no, up_to):
                annotations = []

                while True:
                    a = layers[layer_no][idxs[layer_no]]
                    a_obj = Annotation(
                        start=a['start'],
                        end=a['end'],
                        action=a['action'],
                        colour=a['colour']
                    )

                    if a_obj.start >= up_to: break

                    if layer_no > 0:
                        a_obj.children.extend(get_children(layer_no - 1, a_obj.end))

                    annotations.append(a_obj)

                    # increment cursor only if this even finishes exactly
                    if a_obj.end <= up_to:
                        idxs[layer_no] += 1

                    if idxs[layer_no] >= len(layers[layer_no]): break

                return annotations

            return get_children(num_layers - 1, layers[-1][-1]['end'])

        def discover_contenders(predictions, contenders):
          if len(predictions) == 0: return

          for c in contenders:
              ious = np.array([c.iou(p) for p in predictions])
              idx = np.argmax(ious)
              if ious[idx] == 0.0: continue
              predictions[idx].contenders.append(c)
              predictions[idx].contender_ious.append(ious[idx])

          for p in predictions:
              discover_contenders(p.children, p.contenders)

        result = AnnotationsAsGraph()
        result.ground_truth = [
            Annotation(
                start=a['start'],
                end=a['end'],
                action=a['action'],
                colour=a['colour']
            )
            for a
            in annotations_file['layers'][-1]['annots']
        ]

        # merge groundtruth annotations
        if merge_annots: result.ground_truth = merge_gt(result.ground_truth)

        final_layers = [a['annots'] for a in annotations_file['layers'][:-1] if len(a['annots'])>0]

        # step = annotations_file["duration"]/((2*len(result.ground_truth))-1)
        # final_layers = [[{'start':x, 'end':x+step, 'action':"nothing", 'colour':"teal"} for x in np.linspace(0,annotations_file["duration"],((2*len(result.ground_truth))-1)) ]]
        # final_layers = [[{'start':x, 'end':x+step, 'action':"nothing", 'colour':"teal"} for x in np.linspace(0,annotations_file["duration"],((len(result.ground_truth))+1)) ]]

        result.graph = get_final_layers(final_layers)
        discover_contenders(result.graph, result.ground_truth)
        return result, annotations_file


    @staticmethod
    def combine(predictions):
      def _combine(p):

          if len(p.contenders) == 1:
              return p.contender_ious[0], [(p.contenders[0], p)]

          # highest = np.mean([max(p.contender_ious), *[0.0]*(len(p.contender_ious)-1)])
          highest = max(p.contender_ious)
          highest_gt = np.argmax(p.contender_ious)
          if len(p.children) == 0:
              return highest, [(p.contenders[highest_gt], p)]

          temp = filter(lambda i: len(i.contenders), p.children)
          temp = [_combine(t) for t in temp]
          temp_ious, temp_contenders = [], []
          for i in temp:
              temp_ious.append(i[0])
              temp_contenders.extend(i[1])

          mean = np.mean(temp_ious)

          if highest > mean:
              return highest, [(p.contenders[highest_gt], p)]
          else:
              return mean, temp_contenders

      result = [_combine(p) for p in predictions if len(p.contenders)>0]
      ious, labels = [], []
      for i in result:
          ious.append(i[0])
          labels.extend(i[1])
          i[-1][0][0].gt_match_iou = i[0]

      return np.mean(ious), labels


    @staticmethod
    def apply_colour_and_dictify(tpl):
        (gt, label) = tpl
        label.colour = gt.colour
        return label.to_dict()

    @staticmethod
    def mof(labels, groundtruth, fps, duration):

        pred_bin = np.zeros(int(fps*duration))
        gt_bin = np.zeros(int(fps*duration))

        for label in labels:
            start_frame = int(label.start * fps)
            end_frame = int(label.end*fps)
            pred_bin[start_frame:end_frame] = 1

        for gt in groundtruth:
            start_frame = int(gt.start*fps)
            end_frame = int(gt.end*fps)
            gt_bin[start_frame:end_frame] = 1

        return np.logical_not(np.logical_xor(pred_bin, gt_bin)).mean()

    @staticmethod
    def create_json_layer(objs):
        pass

    @staticmethod
    def run_file(annotations_file, merge_annots=False):
        result, a_file = AnnotationsAsGraph.file_to_obj(annotations_file, merge_annots=merge_annots)

        l = len(os.path.basename(a_file["file"])[:-4].split('_')[-1])
        if l == 2:
            fps = 59.94
        elif l == 3:
            fps = 50.0
        else:
            fps = 60.0
            # raise TypeError("file name not correct")

        mean_iou, labels = AnnotationsAsGraph.combine(result.graph)
        mean_mof = AnnotationsAsGraph.mof(list(map(lambda i:i[-1], labels)), result.ground_truth, fps, a_file["duration"])

        combined_layer = {
            'name': 'Combined',
            'order': 0,
            'annots': list(map(AnnotationsAsGraph.apply_colour_and_dictify, labels))
        }

        gt = a_file['layers'].pop()
        gt['order'] = 1
        # gt['annots'] = [{"start": g.start,
        #                  "end": g.end,
        #                  "colour": g.colour,
        #                  "action": g.action,
        #                  "iou": g.gt_match_iou}
        #                 for g in result.ground_truth]
        a_file['layers'] = [combined_layer, gt]


        return a_file, (mean_iou, mean_mof)


    @staticmethod
    def get_combined_obj(in_file, best_layer=False, merge_annots=False):

        annotations_file = json.load(open(in_file, 'r'))

        if best_layer:
            max_iou, max_mof = 0.0, 0.0
            new_annots = copy.deepcopy(annotations_file)
            gt_layer = copy.deepcopy(annotations_file['layers'][-1])

            for layer in annotations_file['layers'][:-1]:
                if len(layer["annots"]) ==0: continue
                new_annots['layers'] = [layer, annotations_file['layers'][-1]]
                a_file, (iou, mof) = AnnotationsAsGraph.run_file(new_annots, merge_annots=merge_annots)
                max_iou = max(max_iou, iou)
                max_mof = max(max_mof, mof)
            return a_file, (max_iou, max_mof)

        else:
            return AnnotationsAsGraph.run_file(annotations_file, merge_annots=merge_annots)
   
