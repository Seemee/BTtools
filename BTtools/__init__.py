import gzip
import math
from pathlib import Path
from lxml import objectify
from lxml import etree
import copy
import numpy as np 

class BTtools:
    def __init__(self, filename):
        self.filename = filename
        try:
            with gzip.open(filename, 'rb') as f:
                xml=f.read()
            f.close()
        except (IOError, OSError):
            print("Error: Openening file:",filename)
            raise KeyboardInterrupt
        self.obj = objectify.fromstring(xml)

    @staticmethod
    def separation(obj,moves):
        if hasattr(obj,'separation'):
            ob=obj.separation
            cnt=len(ob.state)-1
            moves.append(cnt)
            #print(cnt,end='.')
            BTtools.separation(ob,moves)
        elif hasattr(obj,'separationInfo'):
            lst=str(obj.separationInfo).split(' ')
            for item in lst:
                if item!='0':
                    moves.append(int(item)-1)
        return moves
  
    
    def process(self,problemSelect,ignoreShapes):
        voxelCount={}
        for no,voxel in enumerate(self.obj.shapes.voxel):
            voxelStr=str(voxel)
            identities=sorted(list(set(list(voxelStr))))
            for i,id in enumerate(identities):
                if id.isdigit() and int(id)>0 and int(id)<=9:
                    identities[i]='#'+identities[i]
            #print(identities)
            identities=identities[0:-1]
            for id in reversed(identities):
                voxelStr=voxelStr.replace(id,'%')
            #print(no+1,voxelStr.count('%'))
            voxelCount[no+1]=voxelStr.count('%')

        solutions={}
        problem=self.obj.problems.problem[problemSelect-1]
        state=int(problem.get('state'))
        if state<2:
            print("\nProblem solving for P%d hasn't finished yet, the results will be inaccurate!" % problemSelect)
        print('Processing time %.1f hrs.' % (int(problem.get('time'))/3600) )    
        shapes=[]
        shapeDict={}
        #print(etree.tostring(problem))
        #stop()
        for shape in problem.shapes.shape:
            #print(etree.tostring(shape))
            maxCount=shape.get('count')
            if maxCount!=None:
                minCount=0
            else:
                minCount=0 #shape.get('min')
                maxCount=shape.get('max')
            #print(shape.get('id'),shape.get('max'))
            shapeId=int(shape.get('id'))+1
            shapeDict[shapeId]={}
            shapeDict[shapeId]['name']=self.obj.shapes.voxel[shapeId-1].get('name')
            shapeDict[shapeId]['frequency']=0
            shapeDict[shapeId]['voxels']=voxelCount[shapeId]
            shapeDict[shapeId]['min']=minCount
            shapeDict[shapeId]['max']=maxCount
            for i in range(int(minCount),int(maxCount),1):
                shapes.append(str(shapeId))
        print('Shapes: S',end='')
        print(*shapes,sep=', S')
        pieces=sorted(map(int,set(shapes)))
        piecesMultipleUseList=[0]*(pieces[-1]+1)
        if shapes!=[]:
            cnt=1
            if not hasattr(problem,'solutions'):
                return
            if not hasattr(problem.solutions,'solution'):
                return
            print('Processing',f'{len(problem.solutions.solution):,}','solutions:')
            for sol in problem.solutions.solution:
                if hasMoves:
                    if not hasattr(sol,'pieces'):
                        break
                moves=self.separation(sol,[])
                if moves==[]:
                    moves=[0]
                #level=moves[0]
                totMoves=sum(moves)
                asmNum=sol.get('asmNum')
                lst=str(sol.assembly).split(' ')
                solution=[]
                l=0
                s=0
                print(cnt,end='\r')
                cnt+=1
                piecesMultipleUseTempList=[0]*(pieces[-1]+1)
                while l < len(lst):
                    if lst[l]=='x':
                        l+=1
                        s+=1
                    else:
                        if ignoreShapes==[] or int(shapes[s]) not in ignoreShapes:
                            solution.append(shapes[s])
                            piecesMultipleUseTempList[int(shapes[s])]+=1
                            shapeDict[int(shapes[s])]['frequency']+=1
                        l+=4
                        s+=1
                for pieceNumber in range(pieces[-1]+1):
                    if piecesMultipleUseTempList[pieceNumber]>piecesMultipleUseList[pieceNumber]:
                        piecesMultipleUseList[pieceNumber]=piecesMultipleUseTempList[pieceNumber]
                solution=tuple(solution)
                if asmNum is not None:
                    asmNum=int(asmNum)+1
                if solution in solutions:
                    solutions[solution]['sols']+=1
                    solutions[solution]['asmNum']=asmNum
                    if solutions[solution]['totMoves']<totMoves:
                        solutions[solution]['totMoves']=totMoves
                        solutions[solution]['moves']=moves
                else:
                    solutions[solution]={'sols':1, 'asmNum':asmNum,'totMoves':totMoves,'moves':moves}
        print()
        print('Shape Use:')
        for k,v in shapeDict.items():
            print(' ' if v['name']==None else v['name'],'\tshape: S'+str(k)+'\tvoxels:',v['voxels'],'\tmin:',v['min'],'\tmax:',v['max'],'\tmax use:',piecesMultipleUseList[k],'\tfreq:',v['frequency'])
        print()
        return solutions
    
    @staticmethod
    def bestSolution(solutions):
        minSols=math.inf
        minMoveRange=math.inf
        maxMoveRange=0
        for k,v in solutions.items():
            if v['sols']<minSols:
                minSols=v['sols']
                minAsmNum=v['asmNum']
                moves=v['moves']
                minAssembly=k
                totMoves=v['totMoves']
            elif v['sols']==minSols: # and v['totMoves']<maxMoves:
                minAsmNum=v['asmNum']
                moves=v['moves']
                minAssembly=k
                totMoves=v['totMoves']
            if v['totMoves']<minMoveRange:
                minMoveRange=v['totMoves']
            if v['totMoves']>maxMoveRange:
                maxMoveRange=v['totMoves']
                
        bs= {'minSols':minSols,'totMoves':totMoves,'moves':moves,'minAssembly':minAssembly,'minAsmNum':minAsmNum,'minMoveRange':minMoveRange,'maxMoveRange':maxMoveRange}
        ret = {k: v for k, v in bs.items()}
        return ret
    
    # Subtract p2 from p1
    def subtract(self,p1,p2):
        set2=set()
        problem = self.obj.problems.problem[p2-1]      
        for sol in problem.solutions.solution:
            asmNum=sol.get('asmNum')
            set2.add(asmNum)
        result=[]
        problem = self.obj.problems.problem[p1-1]      
        for sol in problem.solutions.solution:
            asmNum=sol.get('asmNum')
            if not asmNum in set2:
                result.append(sol)
        problem.solutions.solution=result
        return len(result)

    # Compare p1 with p2 -> p1 
    def compare(self,p1,p2):
        set2=set()
        problem = self.obj.problems.problem[p2-1]      
        for sol in problem.solutions.solution:
            asmNum=sol.get('asmNum')
            set2.add(asmNum)
        result=[]
        problem = self.obj.problems.problem[p1-1]      
        for sol in problem.solutions.solution:
            asmNum=sol.get('asmNum')
            if asmNum in set2:
                result.append(sol)
        problem.solutions.solution=result
        return len(result)
    
    def createParts(self,problemSelect):
        obj=self.obj
        voxel=list(str(obj.shapes.voxel[problemSelect-1]))
        identities=sorted(list(set(voxel)))
        for i,id in enumerate(identities):
            if id.isdigit() and int(id)>0 and int(id)<=9:
                identities[i]='#'+identities[i]
        identities=identities[0:-1]
        print('Found %d identities: %s'%(len(identities),', '.join(identities)))
        for no,id in enumerate(identities):
            tmpString=copy.deepcopy(string)
            for i in reversed(identities):
                if i==id:
                    tmpString=tmpString.replace(id,'%')
                else:
                    tmpString=tmpString.replace(i,'_')
            tmpString=tmpString.replace('%',id)
            obj.shapes.append(copy.deepcopy(obj.shapes.voxel[prob]))
            obj.shapes.voxel[-1]=objectify.fromstring('<voxel>'+tmpString+'</voxel>')
            obj.shapes.voxel[-1].set('x',obj.shapes.voxel[prob].get('x'))
            obj.shapes.voxel[-1].set('y',obj.shapes.voxel[prob].get('y'))
            obj.shapes.voxel[-1].set('z',obj.shapes.voxel[prob].get('z'))
            obj.shapes.voxel[-1].set('type',obj.shapes.voxel[prob].get('type'))
            obj.shapes.voxel[-1].set('name','part '+str(no+1))

    def isolate(self,problemSelect):
        obj = self.obj
        prob=problemSelect-1
        problem=obj.problems.problem[prob]
        shapeIds=[]
        for shape in problem.shapes.shape:
            shapeIds.append(int(shape.get('id'))+1)
        newShapeIds=[id for id in range(1,len(shapeIds)+2,1)]
        resultId=int(problem.result.get('id'))
        print('Original shapes: S%s'%(', S'.join(map(str,[resultId]+shapeIds))))
        print('New shape id\'s: S%s'%(', S'.join(map(str,newShapeIds))))
        voxel=[obj.shapes.voxel[resultId]]
        voxel[0].set('name',name+' (P'+str(optimalProblemSelect)+' target shape)')
        for i,id in enumerate(shapeIds):
            problem.shapes.shape[i].set('id',str(i+1))
            voxel.append(obj.shapes.voxel[id-1])
        problem.result.set('id','0')
        obj.problems.problem=[problem]
        obj.shapes.voxel=voxel
        
        
    def bt2pcad(self,problemSelect,scale):
        obj=self.obj
        problem=self.obj.problems.problem[problemSelect-1]
        resultId=int(problem.result.get('id'))
        shapeIds=[resultId+1]
        shapeCounts=[1]
        for shape in problem.shapes.shape:
            shapeCounts.append(int(shape.get('count')))
            shapeIds.append(int(shape.get('id'))+1)
        newShapeIds=[id for id in range(1,len(shapeIds)+2,1)]
        fmt='Shapes found: S'+'S'.join(['%d, ']*len(shapeIds))
        print(fmt[0:-2] % tuple(shapeIds))
        fmt='Shapes selected: S'+'S'.join(['%d, ']*(len(shapeIds)-start+1))
        print(fmt[0:-2] % tuple(shapeIds[start-1:]))
        downScale=int(1/scale)
        voxel=[obj.shapes.voxel[resultId]]
        name=self.obj.problems.problem[problemSelect-1].get('name')
        if name==None:
            name=''
        voxel[0].set('name','%s (P%d target shape)'%(name,problemSelect))
        for i,id in enumerate(shapeIds[1:]):
            problem.shapes.shape[i].set('id',str(i))
            voxel.append(obj.shapes.voxel[id-1])
        problem.result.set('id','0')
        obj.problems.problem=[problem]
        obj.shapes.voxel=voxel
        pcad="""/* 
   BT-Tools Puzzlecad Converter
   This model was generated from P%d of the Burr-Tools file: %s %s
   Feel free to make changes to the model or parameters. 
*/
include <../puzzlecad.scad>
require_puzzlecad_version("2.0");

$burr_scale = 15;
$auto_layout = true;
$burr_inset = 0.2;
$burr_bevel = 0.5;
$unit_beveled = false;
$joint_inset = 0.07;
$plate_width = 240;
$plate_sep = 3;

burr_plate([
""" % (problemSelect,filename,'' if scale==1 else 'at scale %.2f'%scale)
        for no,voxel in enumerate(obj.shapes.voxel[start-1:]):
            name,x,y,z=voxel.get('name'),int(voxel.get('z')),int(voxel.get('y')),int(voxel.get('x'))
            if name==None:
                name=''
            else:
                name='('+name+')'
            identities=sorted(list(set(str(voxel))))
            for i,id in enumerate(identities):
                if id.isdigit() and int(id)>0 and int(id)<=9:
                    identities[i]='#'+identities[i]
            identities=identities[0:-1]
            #print('ids',identities)
            string=str(voxel)
            for num,id in enumerate(identities):
                tmpString=copy.deepcopy(string)
                for i in reversed(identities):
                    if i==id:
                        tmpString=tmpString.replace(id,'x')
                    else:
                        tmpString=tmpString.replace(i,'.')
            tmpString=tmpString.replace('_','.')
            buf= tmpString.encode()
            arr=np.frombuffer(buf,dtype='|S1')
            arr=np.reshape(arr, ( x, y, z ))
            coords=np.array(np.where(arr==b'x'))
            normArr = arr[ np.min(coords[0]):np.max(coords[0]+1) , np.min(coords[1]):np.max(coords[1]+1) , np.min(coords[2]):np.max(coords[2]+1) ]  
            scaledArr=normArr[0::downScale,0::downScale,0::downScale].copy()
            pieceCount=shapeCounts[no+start-1] if copies else 1

            for i in range(pieceCount):
                num = '-'+str(i+1) if pieceCount>1 else ''
                pcad+='\t'+str(['|'.join([ a.tobytes().decode('utf-8') for a in scaledArr[i,:] ]) for i in range(len(scaledArr))])+',\t// S'+str(shapeIds[no+start-1])+num+' '+name+'\n'
        pcad+=']);\n'
        pcad=pcad.replace("'",'"');
        return pcad

    def write(self,extension):
        newXml = etree.tostring(self.obj)
        p=Path(self.filename)
        if extension != '':
            extension='-'+extension
        dst_filename=p.parent.joinpath(p.stem+extension+'.xmpuzzle')
        print('Writing: ',dst_filename)
        with gzip.open(dst_filename, 'wb') as f:
            f.write(newXml)
            f.close()

bt=BTtools()
